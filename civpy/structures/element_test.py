from .element import *


def test_repr():
    e = Element('1', 'a', 'b', None)
    repr(e)


def test_str():
    e = Element('1', 'a', 'b', None)
    assert str(e) == '1'


def test_free():
    e = Element('1', 'a', 'b', None).free()
    a = (e.imx_free, e.imy_free, e.imz_free, e.jmx_free, e.jmy_free, e.jmz_free)
    b = tuple([True] * 6)

    assert a == b


def test_mx_free():
    e = Element('1', 'a', 'b', None).mx_free()
    a = (e.imx_free, e.imy_free, e.imz_free, e.jmx_free, e.jmy_free, e.jmz_free)
    b = (True, False, False, True, False, False)
    assert a == b


def test_my_free():
    e = Element('1', 'a', 'b', None).my_free()
    a = (e.imx_free, e.imy_free, e.imz_free, e.jmx_free, e.jmy_free, e.jmz_free)
    b = (False, True, False, False, True, False)
    assert a == b


def test_mz_free():
    e = Element('1', 'a', 'b', None).mz_free()
    a = (e.imx_free, e.imy_free, e.imz_free, e.jmx_free, e.jmy_free, e.jmz_free)
    b = (False, False, True, False, False, True)
    assert a == b


def test_copy():
    e = Element('1', 'a', 'b', None)
    a = e.copy()

    assert e is not a
    assert repr(e) == repr(a)


def test_sym_elements():
    # No symmetry
    e = Element('1', 'a_p', 'a_x', None)
    p, = e.sym_elements()

    assert (p.name, p.inode, p.jnode) == ('1_p', 'a_p', 'a_x')

    # X symmetry
    e = Element('1', 'a_p', 'a_x', None, symmetry='x')
    p, x = e.sym_elements()

    assert (p.name, p.inode, p.jnode) == ('1_p', 'a_p', 'a_x')
    assert (x.name, x.inode, x.jnode) == ('1_x', 'a_x', 'a_p')

    # Y symmetry
    e = Element('1', 'a_p', 'a_x', None, symmetry='y')
    p, y = e.sym_elements()

    assert (p.name, p.inode, p.jnode) == ('1_p', 'a_p', 'a_x')
    assert (y.name, y.inode, y.jnode) == ('1_y', 'a_y', 'a_xy')

    # XY symmetry
    e = Element('1', 'a_p', 'a_x', None, symmetry='xy')
    p, x, y, xy = e.sym_elements()

    assert (p.name, p.inode, p.jnode) == ('1_p', 'a_p', 'a_x')
    assert (x.name, x.inode, x.jnode) == ('1_x', 'a_x', 'a_p')
    assert (y.name, y.inode, y.jnode) == ('1_y', 'a_y', 'a_xy')
    assert (xy.name, xy.inode, xy.jnode) == ('1_xy', 'a_xy', 'a_y')
