import kivy
kivy.require('1.8.0')

from contextlib import contextmanager
import logging

from kivy.properties import (
    NumericProperty,
    ListProperty,
    BooleanProperty,
)
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
import numpy as np

from .constants import Colours, States

log = logging.getLogger(__name__)

__all__ = ["GridCell", "DrawableGrid"]


class GridCell(Widget):

    state = NumericProperty(States.DEACTIVATED)
    border_state = NumericProperty(States.DEACTIVATED)
    colour = ListProperty(Colours[States.DEACTIVATED])
    border_colour = ListProperty((0, 0, 0, 0))

    def __init__(self, cell_size, coordinates):
        self.coordinates = coordinates
        column_number, row_number = coordinates
        kwargs = {
            "size_hint": [None, None],
            "size": [cell_size, cell_size],
        }
        super(GridCell, self).__init__(**kwargs)
        self.update_canvas()

    def update_canvas(self):
        """ Update the canvas with the current state of the cell

        >>> cell = GridCell(1, (0, 0))
        >>> cell.state = States.FIRST
        >>> cell.update_canvas()
        >>> cell.colour == list(Colours[States.FIRST])
        True
        >>> cell.border_colour
        [0, 0, 0, 0]

        >>> cell.border_state = States.SECOND
        >>> cell.update_canvas()
        >>> cell.border_colour == list(Colours[States.SECOND])
        True
        """
        self.colour = Colours[self.state]
        if self.border_state == States.DEACTIVATED:
            self.border_colour = (0, 0, 0, 0)  # Transparent
        else:
            self.border_colour = Colours[self.border_state]

    def update_parent_cell(self):
        self.parent.update_cells(self.coordinates, self.state)

    def set_state(self, state):
        """ Set the fill state of the cell

        >>> import mock
        >>> cell = GridCell(1, (0, 0))
        >>> cell.parent = mock.Mock()
        >>> cell.set_state(States.FIRST)
        >>> cell.state == States.FIRST
        True
        >>> cell.colour == list(Colours[States.FIRST])
        True
        """
        if hasattr(state, "dtype"):
            assert state.dtype == int, state.dtype
            state = int(state)
        self.state = state
        self.update_canvas()
        self.update_parent_cell()
        log.debug("Set state of {} to {}".format(self, state))

    def set_border_state(self, state):
        """ Set the border state of the cell

        >>> cell = GridCell(1, (0, 0))
        >>> cell.set_border_state(States.FIRST)
        >>> cell.state == States.DEACTIVATED
        True
        >>> cell.colour == list(Colours[States.DEACTIVATED])
        True
        >>> cell.border_colour == list(Colours[States.FIRST])
        True
        """
        if hasattr(state, "dtype"):
            assert state.dtype == int, state.dtype
            state = int(state)
        self.border_state = state
        self.update_canvas()

    def handle_touch(self):
        """ Flip the cell's state between on and off

        >>> import mock
        >>> cell = GridCell(1, (0, 0))
        >>> cell.parent = mock.Mock(selected_state=States.FIRST)
        >>> new_state = cell.handle_touch()
        >>> new_state == cell.state == States.FIRST
        True
        >>> new_state = cell.handle_touch()
        >>> new_state == cell.state == States.DEACTIVATED
        True
        """
        if self.state == self.parent.selected_state:
            new_state = States.DEACTIVATED
        else:
            new_state = self.parent.selected_state
        self.set_state(new_state)
        return new_state

    def on_touch_down(self, evt):
        if not self.collide_point(*evt.pos):
            # Not on this square
            return
        self.handle_touch()

    def on_touch_move(self, evt):
        if not self.collide_point(*evt.pos):
            # Not on this square
            return super(GridCell, self).on_touch_move(evt)
        if self.collide_point(*evt.ppos):
            # Not moved to this square
            return super(GridCell, self).on_touch_move(evt)
        if self.parent.drag_state is None:
            self.parent.drag_state = (
                self.parent.selected_state
                if self.state == States.DEACTIVATED else States.DEACTIVATED
            )
        elif self.parent.drag_state == self.state:
            return
        self.handle_touch()

    def on_touch_up(self, evt):
        if self.parent.drag_state is not None:
            self.parent.drag_state = None

    def __repr__(self):
        return "{}<{}>".format(self.__class__.__name__,
                               ", ".join(str(c) for c in self.coordinates))


class DrawableGrid(RelativeLayout):

    rows = NumericProperty()
    cols = NumericProperty()
    cell_size = NumericProperty(25)
    selected_state = NumericProperty(States.FIRST)
    grids = ListProperty()
    num_grids = NumericProperty(1)

    CELLS_GRID = 0
    GRID_CELL_CLASS = GridCell

    def __init__(self, *args, **kwargs):
        super(DrawableGrid, self).__init__(*args, **kwargs)
        self._cells = None

    def cell_coordinates(self, pos, is_absolute=True):
        """ Determine which cell corresponds to absolute or relative position
        Arguments:
            pos; 2-tuple; Position in pixels
            is_absolute; bool; Is pos an absolute or relative position?

        >>> import mock
        >>> grid = DrawableGrid(cell_size=5)
        >>> grid.to_widget = mock.Mock()
        >>> grid.to_widget.return_value = (111, 111)
        >>> # Returns calculated value
        >>> grid.cell_coordinates((26, 35), is_absolute=False)
        (5, 7)
        >>> grid.to_widget.called
        False
        >>> # Returns mocked value
        >>> grid.cell_coordinates((26, 35))
        (22, 22)
        >>> grid.to_widget.called
        True
        """
        if is_absolute:
            pos = self.to_widget(*pos)
        return (pos[0] // self.cell_size,
                pos[1] // self.cell_size)

    def init_cells(self):
        """ Sets up the grid arrays and the cell widgets

        Simple example:
        >>> grid = DrawableGrid()
        >>> grid.init_cells()
        >>> grid.grids
        [array([], shape=(0, 0), dtype=int64)]
        >>> [g.flags.writeable for g in grid.grids]
        [False]
        >>> grid.cell_widgets
        []

        Example with some cells and multiple grids:
        >>> grid = DrawableGrid(rows=2, cols=1, num_grids=3)
        >>> grid.init_cells()
        >>> grid.grids
        [array([[0, 0]]), array([[0, 0]]), array([[0, 0]])]
        >>> [g.flags.writeable for g in grid.grids]
        [False, False, False]
        >>> grid.cell_widgets
        [[GridCell<0, 0>], [GridCell<0, 1>]]

        Check that overwriting is forbidden
        >>> grid.init_cells()
        Traceback (most recent call last):
        RuntimeError: Cells already initialised!
        """
        if self._cells is not None:
            raise RuntimeError("Cells already initialised!")
        self._setup_cell_widgets()
        self._cells = np.zeros(dtype=int, shape=(self.cols, self.rows))
        self.grids = [self._cells]
        for num in range(1, self.num_grids):
            self.grids.append(self._cells.copy())
        for grid in self.grids:
            grid.setflags(write=False)
        self.drag_state = None

    def _setup_cell_widgets(self):
        self.cell_widgets = []
        for row_number in xrange(self.rows):
            row = []
            for column_number in xrange(self.cols):
                cell = self.GRID_CELL_CLASS(
                    self.cell_size, (column_number, row_number))
                cell.y = (row_number) * self.cell_size
                cell.x = (column_number) * self.cell_size
                row.append(cell)
            self.cell_widgets.append(row)
        with self.canvas:
            for row in self.cell_widgets:
                for cell in row:
                    self.add_widget(cell)

    @contextmanager
    def _writable_grid(self, index):
        """Set self.grids[index] to be writable, then unset it"""
        grid = self.grids[index]
        try:
            grid.setflags(write=True)
            yield
        finally:
            grid.setflags(write=False)
            return

    def on_cells_updated(self):
        """This is a hook to update things when the cells have been updated"""
        pass

    @property
    def writable_cells(self):
        """
        Usage:
        >>> grid = DrawableGrid()
        >>> grid.init_cells()
        >>> grid.cells.flags.writeable
        False
        >>> with grid.writable_cells:
        ...     grid.cells.flags.writeable
        True
        >>> grid.cells.flags.writeable
        False
        """
        return self._writable_grid(index=self.CELLS_GRID)

    def update_cells(self, coordinates, state):
        """ Set cell state at coordinates.
        Arguments:
            coordinates; 2-tuple; Cell coordinates to update
            state; int; New state for the cell

        >>> grid = DrawableGrid(rows=2, cols=1)
        >>> grid.init_cells()
        >>> grid.cells
        array([[0, 0]])
        >>> grid.update_cells((0, 0), 1)
        >>> grid.cells
        array([[1, 0]])
        """
        with self.writable_cells:
            self._cells[coordinates] = state
        self.on_cells_updated()

    def set_cell_state(self, cell, y, x):
        cell.set_state(self.cells[y, x])

    def update_cell_widgets(self):
        """ Set each cell widget's state according to the state of the np grid

        >>> grid = DrawableGrid(rows=2, cols=1)
        >>> grid.init_cells()
        >>> grid.cells = np.array([[1, 2]])
        >>> grid.update_cell_widgets()
        >>> grid.cell_widgets[0][0].state
        1
        >>> grid.cell_widgets[1][0].state
        2
        """
        for x, row in enumerate(self.cell_widgets):
            for y, cell in enumerate(row):
                self.set_cell_state(cell, y, x)

    def clear_grid(self, index):
        """ Replace the chosen grid with a zero grid of the same shape
        Arguments:
            index; bool; Index of the grid to update

        >>> grid = DrawableGrid(rows=2, cols=1)
        >>> grid.init_cells()
        >>> grid.cells = np.array([[1, 2]])
        >>> grid.clear_grid(0)
        >>> grid.cells
        array([[0, 0]])
        """
        new_grid = np.zeros_like(self.grids[index])
        if index == self.CELLS_GRID:
            # cells property does everything we need
            self.cells = new_grid
        else:
            new_grid.setflags(write=False)
            self.grids[index] = new_grid

    def clear_grid_for_event(self, grid_index, evt):
        """ This is designed to be subclassed, so that only part of the grid
            can be cleared, if so desired. """
        return self.clear_grid(grid_index)

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cells):
        """
        Cell values can be set here. This will update the related widgets.
        """
        if hasattr(cells, "copy"):
            # Assume cells is a numpy array
            cells = cells.copy()
        else:
            cells = np.array(cells)
        cells.setflags(write=False)
        assert cells.ndim == 2, cells.ndim
        assert cells.shape == self._cells.shape, "{} != {}".format(
            cells.shape, self._cells.shape)
        assert cells.dtype == self._cells.dtype, "{} != {}".format(
            cells.dtype, self._cells.dtype)
        self._cells = cells
        self.grids[self.CELLS_GRID] = cells
        self.on_cells_updated()
        self.update_cell_widgets()
