import attr

@attr.s
class Speaker:
    video = attr.ib()
    identity = attr.ib()

@attr.s
class Config:
    targets = attr.ib()
    help = attr.ib(default=False)
    debug = attr.ib(default=False)
    show_frame = attr.ib(default=False)
    framerate = attr.ib(default=25)
    face_updates = attr.ib(default=5)
    face_detector = attr.ib(default='hog')
    face_detect_subsample = attr.ib(default=2)
    face_detect_area = attr.ib(default=None)
