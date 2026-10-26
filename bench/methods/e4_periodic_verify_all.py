"""E4 upper bound: same verification applied to ALL images (no prefilter)."""
from proto_e4.method import E4


class E4All(E4):
    name = 'e4_periodic_verify_all'
    verify_all = True


METHOD = E4All()
