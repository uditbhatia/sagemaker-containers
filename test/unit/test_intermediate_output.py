# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import contextlib
import importlib
import os
import sys
import tarfile
import textwrap

from mock import call, mock_open, patch
import pytest
from six import PY2

from sagemaker_containers import _errors, _modules, _params, _intermediate_output

builtins_open = '__builtin__.open' if PY2 else 'builtins.open'

REGION = 'us-west'


def test_accept_file_output_no_process():
    intemediate_sync = _intermediate_output.start_intermediate_folder_sync(
        'file://my/favorite/file', REGION)
    assert intemediate_sync is None


def test_wrong_output():
    with pytest.raises(ValueError) as e:
        _intermediate_output.start_intermediate_folder_sync('tcp://my/favorite/url', REGION)
    assert 'Expecting \'s3\' scheme' in str(e)


def test_daemon_process():
    intemediate_sync = _intermediate_output.start_intermediate_folder_sync('s3://mybucket/', REGION)
    assert intemediate_sync.daemon is True


def test_files_are_preserved():
    pass


def test_delete_files():
    pass


def test_s3_upload():
    pass


def test_multipart_upload():
    pass


def test_every_modification_triggers_upload():
    pass

