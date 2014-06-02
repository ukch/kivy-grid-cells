import kivy
kivy.require('1.8.0')

from kivy.core.window import Window
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout

from kivy_grid_cells import DrawableGrid


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
