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

    def activate(self):
        self.active = True
        self.update_canvas()
        self.update_parent_cell()

    def deactivate(self):
        self.active = False
        self.update_canvas()
        self.update_parent_cell()

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

        self._setup_cells()
        self.cells = np.zeros(dtype=int, shape=(self.cols, self.rows))
        self.drag_state = None

    def _setup_cells(self):
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

    def update_cells(self, coordinates, state):
        self.cells[coordinates] = state

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
