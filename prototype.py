from contextlib import contextmanager
import logging as log # TODO proper logger config

import numpy as np

import kivy
kivy.require('1.8.0')

from kivy.core.window import Window
from kivy.app import App
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.floatlayout import FloatLayout

from kivy.properties import (
    NumericProperty,
    ObjectProperty,
    ListProperty,
    ReferenceListProperty,
    BooleanProperty,
)


class State(object):
    DEACTIVATED = 0
    ACTIVATED = 1

    @classmethod
    def get(cls, active):
        if active:
            return cls.ACTIVATED
        return cls.DEACTIVATED

class Colours(object):
    ACTIVATED = (1, 1, 1, 1)
    DEACTIVATED = (0.5, 0.5, 0.5, 1)


class GridCell(Widget):

    active = BooleanProperty(False)
    colour = ListProperty(Colours.DEACTIVATED)

    def __init__(self, cell_size, coordinates):
        self.coordinates = coordinates
        column_number, row_number = coordinates
        kwargs = {
            "size_hint": [None, None],
            "size": [cell_size, cell_size],
            "x": (column_number + 1) * cell_size,
            "y": (row_number + 1) * cell_size
        }
        super(GridCell, self).__init__(**kwargs)
        self.update_canvas()

    def update_canvas(self):
        self.colour = (Colours.ACTIVATED if self.active
                       else Colours.DEACTIVATED)

    def update_parent_cell(self):
        self.parent.update_cells(self.coordinates, State.get(self.active))

    def set_state(self, state):
        if state == State.ACTIVATED:
            self.active = True
        else: # For now, assume only two states
            self.active = False
        self.update_canvas()
        self.update_parent_cell()

    def activate(self):
        self.set_state(State.ACTIVATED)

    def deactivate(self):
        self.set_state(State.DEACTIVATED)

    def handle_touch(self):
        if self.active:
            self.deactivate()
            log.info("Deactivated {}".format(self))
        else:
            self.activate()
            log.info("Activated {}".format(self))

    def on_touch_down(self, evt):
        if not self.collide_point(*evt.pos):
            # Not on this square
            return
        self.handle_touch()

    def on_touch_move(self, evt):
        if not self.collide_point(*evt.pos):
            # Not on this square
            return
        if self.collide_point(*evt.ppos):
            # Not moved to this square
            return
        drag_state = State.ACTIVATED if self.active else State.DEACTIVATED
        if self.parent.drag_state is None:
            self.parent.drag_state = drag_state
        elif self.parent.drag_state != drag_state:
            return
        self.handle_touch()

    def on_touch_up(self, evt):
        if self.parent.drag_state is not None:
            self.parent.drag_state = None

    def __repr__(self):
        return "{}<{}>".format(self.__class__.__name__,
                               ", ".join(str(c) for c in self.coordinates))


class DrawableGrid(RelativeLayout):

    rows = NumericProperty(10)
    cols = NumericProperty(10)
    cell_size = NumericProperty(25)

    def __init__(self, *args, **kwargs):
        super(DrawableGrid, self).__init__(*args, **kwargs)

        self._setup_cell_widgets()
        self._cells = np.zeros(dtype=int, shape=(self.cols, self.rows))
        self._cells.setflags(write=False)
        self.drag_state = None

    def _setup_cell_widgets(self):
        self.cell_widgets = []
        for row_number in xrange(self.rows):
            row = []
            for column_number in xrange(self.cols):
                cell = GridCell(self.cell_size, (column_number, row_number))
                cell.y = (row_number + 1) * self.cell_size
                cell.x = (column_number + 1) * self.cell_size
                row.append(cell)
            self.cell_widgets.append(row)
        with self.canvas:
            for row in self.cell_widgets:
                for cell in row:
                    self.add_widget(cell)

    @property
    @contextmanager
    def writable_cells(self):
        """Set self._cells to be writable, then unset it"""
        try:
            self._cells.setflags(write=True)
            yield
        finally:
            self._cells.setflags(write=False)
            return

    def update_cells(self, coordinates, state):
        with self.writable_cells:
            self._cells[coordinates] = state

    def update_cell_widgets(self):
        cells = self.cells
        for x, row in enumerate(self.cell_widgets):
            for y, cell in enumerate(row):
                cell.set_state(cells[y, x])

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
        assert cells.shape == self._cells.shape, "{} != {}".format(
            cells.shape, self._cells.shape)
        self._cells = cells
        self.update_cell_widgets()

    @property
    def rows_adjusted(self):
        return self.rows * self.cell_size

    @property
    def cols_adjusted(self):
        return self.cols * self.cell_size


class GridPrototype(App):

    def build(self):
        self.root = FloatLayout()
        self.grid = DrawableGrid(rows=10, cols=15, size_hint=(None, None))
        self.root.add_widget(self.grid)
        return self.root

    def on_start(self):
        # TODO can we calculate this in the kv file?
        def refresh_grid_position(*args):
            self.grid.center = self.root.center

        Window.bind(on_resize=refresh_grid_position)
        refresh_grid_position()


if __name__ == '__main__':
    GridPrototype().run()
