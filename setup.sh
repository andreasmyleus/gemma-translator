#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

cd "$(dirname "$0")"

# Pick an interpreter that satisfies the 3.10+ requirement; the system `python3`
# is often older (e.g. 3.8 on macOS with an old Homebrew default).
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" > /dev/null 2>&1 &&
           "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
            PYTHON="$candidate"
            break
        fi
    done
fi
if [ -z "$PYTHON" ]; then
    echo "Error: no Python 3.10+ interpreter found. Install one or set PYTHON=/path/to/python3."
    exit 1
fi

echo "Creating virtual environment with $($PYTHON --version)..."
"$PYTHON" -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing requirements..."
pip install --extra-index-url https://pypi.org/simple/ -r backend/requirements.txt

echo "========================================="
echo "Setup complete!"
echo "Run ./download_model.sh to download the model."
echo "Run ./start.sh to start the servers."
echo "========================================="
