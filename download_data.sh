# MIT License
#
# Copyright (c) 2017, Andrew Warrington.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# configure_docker.sh
# This script configures the environment inside a docker container, 
# (re-)downloading the git repository containing the code and curl-ing 
# the data from the website and placing it in the correct place.
# Although this is a bit of a luddite way of performing this, it means we
# do not have to modify the pre-build docker base image (ufoym/deepo).

# Script downloads the preformatted data from my website and unzips it such
# that it is in the right place for the subsequent scripts to find and use it.
# Requires cURL and unzip.

# Download the Kasthuri data from the wesbite.
curl -O http://www.robots.ox.ac.uk/~andreww/data/kasthuri_data.zip
unzip kasthuri_data.zip
