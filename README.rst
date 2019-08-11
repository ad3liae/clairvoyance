.. -*- mode: rst -*-
Clairvoyance: Concurrent, CLI Lip-reader
========================================

Copyright (C) 2019 Ken-ya and Takahiro Yoshimura. All rights reserved.
Licensed under the terms of GNU General Public License Version 3 or later. See ``COPYING.txt`` for details.

Clairvoyance is a concurrent lip reader. It is capable to read speakers are speaking in the given video stream, showing the threads of conversations.

Quickstart
==========

You need: Python 3.6, ffmpeg (with ffprobe)

*NB: Currently it requires to be installed in development mode.*

.. code-block:: shell

  $ python3.6 -m venv ~/ve/cv
  $ git clone https://github.com/monolithworks/clairvoyance wc
  $ cd wc
  $ ~/ve/cv/bin/pip install -e .
  $ ~/ve/cv/bin/pip install -r requirements.txt

  $ ~/ve/cv/bin/clairvoyance --debug --show-frame --face-detect-subsample=1 /path/to/some/movie.mp4
  ...
  DEBUG:FaceDetector:known_0: 70: recognized
  DEBUG:FaceDetector:known_1: 70: recognized
  DEBUG:FaceDetector:known_2: 70: recognized
  DEBUG:FaceDetector:known_3: 70: recognized
  ...
  DEBUG:FaceRecognitionTask:Faces detected: 4 (18.67 sec.).
  known_0: bin re at z four now (6.60 sec)
  known_1: bin bred ich c five son (3.63 sec)
  known_2: ben are in f see son (3.72 sec)
  known_3: h b o n o one son (6.10 sec)


Bugs
====

Lots; slow, arcane, have to be stopped with C-c after detection, and so on. Sorry for that.
