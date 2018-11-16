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
from sagemaker_containers import _env, _params


intermediate_path = _env.output_intermediate_dir  # type: str
failure_file_path = os.path.join(_env.output_dir, 'failure')  # type: str
success_file_path = os.path.join(_env.output_dir, 'success')  # type: str
tmp_dir_path = os.path.join(intermediate_path, '.tmp.sagemaker_s3_sync')  # type: str


def _timestamp():
    """Return a timestamp with millisecond precision."""
    moment = time.time()
    moment_ms = repr(moment).split('.')[1][:3]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_ms), time.gmtime(moment))


def _upload_to_s3(s3_uploader, relative_path, filename):
    file_path = os.path.join(tmp_dir_path, relative_path, filename)
    print('Upload to s3: {}'.format(file_path))
    # We know the exact length of the timestamp (24) we are adding to the filename
    key = os.path.join(s3_uploader['key_prefix'], relative_path, filename[24:])
    s3_uploader['transfer'].upload_file(file_path, s3_uploader['bucket'], key)
    print('Uploaded to s3: {}'.format(key))


def _move_file(relative_path, file):
    print('moving file : {}'.format(os.path.join(intermediate_path, relative_path, file)))
    new_filename = '{}.{}'.format(_timestamp(), file)
    shutil.move(os.path.join(intermediate_path, relative_path, file),
                os.path.join(tmp_dir_path, relative_path, new_filename))
    return new_filename


def _watch(inotify, watchers, watch_flags, s3_uploader):
    print('Starting to listen for updates')
    # initialize a thread pool with 1 worker
    # to be used for uploading files to s3 in a separate thread
    executor = ThreadPoolExecutor(max_workers=1)

    last_pass_done = False
    stop_file_exists = False

    # after we see stop file do one additional pass to make sure we didn't miss anything
    while not last_pass_done:
        # wait for any events in the directory for 1 sec and then re-check exit conditions
        for event in inotify.read(timeout=1000):
            print()
            print(event)
            for flag in flags.from_mask(event.mask):
                if flag is flags.ISDIR:
                    for folder, dirs, files in os.walk(os.path.join(intermediate_path, event.name)):
                        wd = inotify.add_watch(folder, watch_flags)
                        relative_path = os.path.relpath(folder, intermediate_path)
                        watchers[wd] = relative_path
                        os.makedirs(os.path.join(tmp_dir_path,
                                                 relative_path))
                        for file in files:
                            filename = _move_file(relative_path, file)
                            executor.submit(_upload_to_s3, s3_uploader, relative_path, filename)
                elif flag is flags.CLOSE_WRITE:
                    filename = _move_file(watchers[event.wd], event.name)
                    executor.submit(_upload_to_s3, s3_uploader, watchers[event.wd], filename)

        last_pass_done = stop_file_exists
        stop_file_exists = os.path.exists(success_file_path) or os.path.exists(failure_file_path)

    # wait for all the s3 upload tasks to finish and shutdown the executor
    print('Not listening.')
    executor.shutdown(wait=True)
    print('==The End==')


def start_intermediate_folder_sync(s3_output_location, region):
    """We need to initialize intermediate folder behavior only if the directory doesn't exists yet.
    If it does - it indicates that platform is taking care of syncing files to S3
    and container should not interfere.
    """
    print('s3_output_location: {}'.format(s3_output_location))
    print('region: {}'.format(region))
    print('os.path.exists(intermediate_path): {}'.format(os.path.exists(intermediate_path)))
    if not s3_output_location or os.path.exists(intermediate_path):
        print('Could not initialize intermediate folder sync to s3.')
        return None

    # create intermediate and intermediate_tmp directories
    os.makedirs(intermediate_path)
    os.makedirs(tmp_dir_path)

    # configure unique s3 output location similar to how SageMaker platform does it
    s3_output_location = os.path.join(s3_output_location, os.environ.get('TRAINING_JOB_NAME', None),
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
