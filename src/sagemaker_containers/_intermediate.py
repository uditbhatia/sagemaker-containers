# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from concurrent.futures import ThreadPoolExecutor
from inotify_simple import INotify, flags
from multiprocessing import Process
import os
import shutil
from six.moves.urllib.parse import urlparse
import time

import boto3
from boto3.s3.transfer import S3Transfer
from sagemaker_containers import _env, _logging

logger = _logging.get_logger()

intermediate_path = _env.output_intermediate_dir  # type: str
failure_file_path = os.path.join(_env.output_dir, 'failure')  # type: str
success_file_path = os.path.join(_env.output_dir, 'success')  # type: str
tmp_dir_path = os.path.join(intermediate_path, '.tmp.sagemaker_s3_sync')  # type: str


def _timestamp():
    """Return a timestamp with microsecond precision."""
    moment = time.time()
    moment_us = repr(moment).split('.')[1]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_us), time.gmtime(moment))


def _upload_to_s3(s3_uploader, relative_path, file_path, filename):
    try:
        #print('Upload to s3: {}'.format(file_path))
        key = os.path.join(s3_uploader['key_prefix'], relative_path, filename)
        s3_uploader['transfer'].upload_file(file_path, s3_uploader['bucket'], key)
        #print('Uploaded to s3: {}'.format(key))

        # delete the original file
        #print('Deleting file after upload: {}'.format(file_path))
        os.remove(file_path)
    except FileNotFoundError:
        # Broken link or deleted
        pass
    except Exception:
        logger.exception('Failed to upload file to s3.')

        # delete the original file
        #print('Deleting file after upload: {}'.format(file_path))
        os.remove(file_path)


def _move_file(executor, s3_uploader, relative_path, filename):
    try:
        src = os.path.join(intermediate_path, relative_path, filename)
        dst = os.path.join(tmp_dir_path, relative_path, '{}.{}'.format(_timestamp(), filename))
        #print('copying file : "{}" to "{}'.format(src, dst))
        shutil.copy2(src, dst)
        executor.submit(_upload_to_s3, s3_uploader, relative_path, dst, filename)
    except FileNotFoundError:
        # Broken link or deleted
        pass
    except Exception:
        logger.exception('Failed to copy file to the temporarily directory.')


def _watch(inotify, watchers, watch_flags, s3_uploader):
    #print('Starting to listen for updates')
    # initialize a thread pool with 1 worker
    # to be used for uploading files to s3 in a separate thread
    executor = ThreadPoolExecutor(max_workers=1)

    last_pass_done = False
    stop_file_exists = False

    # after we see stop file do one additional pass to make sure we didn't miss anything
    while not last_pass_done:
        # wait for any events in the directory for 1 sec and then re-check exit conditions
        for event in inotify.read(timeout=1000):
            #print()
            #print(event)
            for flag in flags.from_mask(event.mask):
                if flag is flags.ISDIR:
                    for folder, dirs, files in os.walk(os.path.join(intermediate_path, event.name)):
                        wd = inotify.add_watch(folder, watch_flags)
                        relative_path = os.path.relpath(folder, intermediate_path)
                        watchers[wd] = relative_path
                        os.makedirs(os.path.join(tmp_dir_path,
                                                 relative_path))
                        for file in files:
                            _move_file(executor, s3_uploader, relative_path, file)
                elif flag is flags.CLOSE_WRITE:
                    _move_file(executor, s3_uploader, watchers[event.wd], event.name)

        last_pass_done = stop_file_exists
        stop_file_exists = os.path.exists(success_file_path) or os.path.exists(failure_file_path)

    # wait for all the s3 upload tasks to finish and shutdown the executor
    #print('Not listening.')
    executor.shutdown(wait=True)
    #print('==The End==')


def start_intermediate_folder_sync(s3_output_location, region):
    """We need to initialize intermediate folder behavior only if the directory doesn't exists yet.
    If it does - it indicates that platform is taking care of syncing files to S3
    and container should not interfere.
    """
    #print('s3_output_location: {}'.format(s3_output_location))
    #print('region: {}'.format(region))
    #print('os.path.exists(intermediate_path): {}'.format(os.path.exists(intermediate_path)))
    if not s3_output_location or os.path.exists(intermediate_path):
        logger.debug('Could not initialize intermediate folder sync to s3.')
        return None

    # create intermediate and intermediate_tmp directories
    os.makedirs(intermediate_path)
    os.makedirs(tmp_dir_path)

    # configure unique s3 output location similar to how SageMaker platform does it
    s3_output_location = os.path.join(s3_output_location, os.environ.get('TRAINING_JOB_NAME', ''),
                                      'output', 'intermediate')
    url = urlparse(s3_output_location)
    if url.scheme != 's3':
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))

    # create s3 transfer client
    client = boto3.client('s3', region)
    s3_transfer = S3Transfer(client)
    s3_uploader = {
        'transfer': s3_transfer,
        'bucket': url.netloc,
        'key_prefix': url.path.lstrip('/'),
    }

    # Add intermediate folder to the watch list
    inotify = INotify()
    watch_flags = flags.CLOSE_WRITE | flags.CREATE
    watchers = {}
    wd = inotify.add_watch(intermediate_path, watch_flags)
    watchers[wd] = ''

    # start subrocess to sync any files from intermediate folder to s3
    p = Process(target=_watch, args=[inotify, watchers, watch_flags, s3_uploader])
    # Make the process daemonic as a safety switch to prevent training job from hanging forever
    # in case if something goes wrong and main container process exists in an unexpected
    p.daemon = True
    p.start()
    return p
