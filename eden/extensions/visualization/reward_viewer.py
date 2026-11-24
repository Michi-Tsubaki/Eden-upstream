"""Live reward plots and interactive disturbances during a policy rollout."""

import time
from collections import deque
from queue import SimpleQueue

import genesis as gs
import numpy as np
import torch
from viser import uplot

from eden.extensions.visualization.viser import ViserViewer


class RewardViewer(ViserViewer):
    def __init__(self, env, *, host="localhost", port=8080):
        if not env.unwrapped.reward_manager.active_terms:
            raise ValueError("Reward plots require a task with at least one active reward term with nonzero weight.")
        super().__init__(env.unwrapped, host=host, port=port)
        self._step_env = env.step

    def _setup_gui(self):
        manager = self.env.reward_manager

        gui = self.server.gui
        self.playing = gui.add_checkbox("Play", initial_value=False)
        self.env_index = gui.add_number(
            "Environment", 0, min=0, max=self.env.num_envs - 1, step=1, visible=self.env.num_envs > 1
        )
        self._history = deque(maxlen=200)
        self._last_env = None
        self._impulses = SimpleQueue()
        links = {
            f"{name}/{link.name}": link.idx
            for name, entity in self.env.entities.items()
            if entity.solver is self.env.rigid_solver
            for link in entity.links
            if not link.is_fixed
        }
        if links:
            with gui.add_folder("Disturbance"):
                target = gui.add_dropdown("Link", tuple(links))
                impulse = gui.add_vector3("World impulse (N s)", (1.0, 0.0, 0.0), step=0.01)
                push = gui.add_button("Apply impulse")

                @push.on_click
                def _(_):
                    self._impulses.put((self.env_index.value, links[target.value], impulse.value))

        self._weights = []
        self._plots = {}
        with gui.add_folder("Weighted rewards per second"):
            for name, term in manager.iter_terms():
                with gui.add_folder(name) as folder:
                    self._weights.append((term, gui.add_number("Weight", term.weight, step=0.001)))
                    plot = gui.add_uplot(
                        data=(np.empty(0), np.empty(0)),
                        series=(uplot.Series(label="Time (s)"), uplot.Series(label="Rate", stroke="#3b82f6")),
                        scales={"x": uplot.Scale(time=False)},
                        aspect=2.8,
                    )
                self._plots[name] = folder, plot

    def step(self, action, *, pace=True):
        if not self._built:
            self.update()
        while True:
            index = self.env_index.value
            if self._last_env != index:
                self._history.clear()
                self._last_env = index
                for name, (folder, plot) in self._plots.items():
                    folder.label = name
                    plot.data = (np.empty(0), np.empty(0))
            if self.playing.value:
                break
            time.sleep(0.02)

        start = time.monotonic()
        episode_step = int(self.env.episode_length_buf[index]) + 1
        for term, control in self._weights:
            term.weight = control.value
        while not self._impulses.empty():
            env_index, link_index, impulse = self._impulses.get_nowait()
            force = torch.tensor(impulse, device=self.env.device, dtype=gs.tc_float) / self.env.env_options.sim_dt
            self.env.rigid_solver.apply_links_external_wrench(
                force=force.reshape(1, 1, 3),
                links_idx=[link_index],
                envs_idx=[env_index],
                ref=gs.link_ref_frame.link_COM,
                local=False,
            )

        result = self._step_env(action)
        if episode_step == 1:
            self._history.clear()
        rates = self.env.reward_manager.step_reward_rates[index].tolist()
        self._history.append((episode_step * self.env.dt, *rates))
        times, *values = np.asarray(self._history).T
        with self.server.atomic():
            self.update()
            for (name, (folder, plot)), series in zip(self._plots.items(), values, strict=True):
                folder.label = f"{name}  {series[-1]:.4g}"
                plot.data = (times, series)
        if pace:
            time.sleep(max(0.0, self.env.dt - (time.monotonic() - start)))
        return result
