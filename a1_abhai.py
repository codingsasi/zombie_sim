"""
COSC 3117 Assignment 1: Zombie outbreak, built from mesa_web_wealth.py.

Part 1 rules
    One agent type, OutbreakAgent, with isZombie, shots_left = 15 and dead.
    10% of agents start as zombies (num_humans = 90, num_zombies = 10 by default).
    All agents move randomly.
    Each step has two phases. First every agent moves, then every agent interacts with
    the others in its new cell (shooting for humans, infecting for zombies), in random
    order. Shooting therefore happens for every human, however it moved.
    Zombie: if it shares a cell with non-zombies it turns one of them (at random)
        into a zombie. It has a 50% chance of dropping some of its ammo to a human
        in the same cell. Zombies cannot shoot.
    Human: if it shares a cell with a zombie it has a 50% chance of shooting one.
        A successful shot sets dead = True. Humans can run out of ammo.
        A dead zombie does not move and no longer infects.
    The gini function is replaced by a count of humans left at each step.

Part 2 features
    FEATURE 1, immunity: with immunity = True each starting human has a chance of being
        immune (immunity_chance, by default 1 in N where N is the starting number of humans
        plus zombies like in Last of Us :) ). Once one human is immune no other human can be
        immune by chance, so a run has at most one natural immune human, or none. A zombie
        cannot bite an immune human. Each step an immune human shares its immunity with every
        human within `immunity_radius` cells (its own cell included) who is not a zombie, and those
        humans then share it in the same way. immunity = False gives the plain Part 1
        behaviour.
    FEATURE 2, immune humans can be killed: with immune_can_die = True a zombie that shares
        a cell with an immune human who has no ammo left (shots_left = 0) kills that human
        (dead = True, the human stays on the grid and counts in DeadHumans). Immune humans
        who still have ammo cannot be killed. This only has an effect when immunity is on.

Run
    solara run a1_abhai.py     web visualisation
    python a1_abhai.py            batch run, writes a1_results.csv next to this file
"""

import os

import mesa
import pandas as pd

from mesa.visualization import SolaraViz, make_plot_component, make_space_component

SHOTS_START = 15
HIT_CHANCE = 0.5
DROP_CHANCE = 0.5
DROP_MAX = 3  # a zombie drops between 1 and DROP_MAX shots at a time


class OutbreakAgent(mesa.Agent):
    """A human or a zombie, depending on isZombie."""

    def __init__(self, model, is_zombie=False):
        super().__init__(model)
        self.isZombie = is_zombie
        self.shots_left = model.shots_start
        self.dead = False
        # Feature 1: an immune human cannot be bitten and can pass immunity on.
        self.immune = False
        # Step on which this agent was bitten, so it does not act twice in that step.
        self.turned_on = None

    @property
    def is_live_zombie(self):
        return self.isZombie and not self.dead

    @property
    def is_human(self):
        return not self.isZombie and not self.dead

    def move_phase(self):
        if self.dead:
            return

        if self.is_live_zombie:
            self.move_random()
        elif self.is_human:
            self.move_random()

    def interact_phase(self):
        # An agent bitten earlier in this phase does not act again in the same step.
        if self.dead or self.turned_on == self.model.steps:
            return
        if self.is_live_zombie:
            self.infect_and_drop_ammo()
        else:
            if self.immune:
                self.share_immunity()
            self.shoot()

    def turn(self):
        self.isZombie = True
        self.turned_on = self.model.steps

    # Movement

    def neighbour_cells(self):
        return self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

    def move_random(self):
        self.model.grid.move_agent(self, self.random.choice(self.neighbour_cells()))

    # Interaction

    def cellmates(self):
        return [a for a in self.model.grid.get_cell_list_contents([self.pos]) if a is not self]

    def infect_and_drop_ammo(self):
        humans = [a for a in self.cellmates() if a.is_human]
        if not humans:
            return

        # One action per step on one human chosen at random. Immune humans cannot be
        # bitten (feature 1), but with feature 2 on an immune human with no ammo can be killed.
        targets = [
            a
            for a in humans
            if not a.immune or (self.model.immune_can_die and a.shots_left == 0)
        ]
        if targets:
            victim = self.random.choice(targets)
            if victim.immune:
                victim.dead = True
            else:
                victim.turn()

        # 50% chance of dropping some ammo to a human still in the cell.
        humans = [a for a in humans if a.is_human]
        if humans and self.shots_left > 0 and self.random.random() < DROP_CHANCE:
            amount = min(self.random.randint(1, DROP_MAX), self.shots_left)
            self.shots_left -= amount
            self.random.choice(humans).shots_left += amount

    def share_immunity(self):
        """Feature 1: immunity spreads to nearby humans who are not zombies."""
        nearby = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=True, radius=self.model.immunity_radius
        )
        for a in nearby:
            if a.is_human:
                a.immune = True

    def shoot(self):
        zombies = [a for a in self.cellmates() if a.is_live_zombie]
        if not zombies or self.shots_left <= 0:
            return
        self.shots_left -= 1
        if self.random.random() < HIT_CHANCE:
            self.random.choice(zombies).dead = True


# Replaces compute_gini: the counts tracked at each step.

def count_humans(model):
    return sum(1 for a in model.agents if a.is_human)


def count_immune(model):
    return sum(1 for a in model.agents if a.is_human and a.immune)


def count_zombies(model):
    return sum(1 for a in model.agents if a.is_live_zombie)


def count_dead_zombies(model):
    return sum(1 for a in model.agents if a.isZombie and a.dead)


def count_dead_humans(model):
    return sum(1 for a in model.agents if not a.isZombie and a.dead)


class OutbreakModel(mesa.Model):
    """Outbreak on a torus grid. Several agents can share a cell."""

    def __init__(
        self,
        num_humans=90,
        num_zombies=10,
        width=20,
        height=20,
        immunity=False,
        immune_can_die=False,
        shots_start=SHOTS_START,
        immunity_radius=1,
        immunity_chance=None,
        seed=None,
    ):
        # The web form passes the seed as text once it has been edited.
        super().__init__(seed=int(seed) if seed not in (None, "") else None)
        self.immune_can_die = immune_can_die
        self.shots_start = shots_start
        self.immunity_radius = immunity_radius
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Humans": count_humans,
                "Immune": count_immune,
                "Zombies": count_zombies,
                "DeadZombies": count_dead_zombies,
                "DeadHumans": count_dead_humans,
            }
        )
        chance = immunity_chance if immunity_chance is not None else 1 / (num_humans + num_zombies)
        natural_immune_exists = False
        for i in range(num_humans + num_zombies):
            agent = OutbreakAgent(self, is_zombie=i < num_zombies)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(agent, (x, y))
            # Feature 1: chance that a starting human is immune, until one human is. The
            # number is always drawn, so the same seed gives the same layout with the
            # feature on or off.
            roll = self.random.random()
            if immunity and not natural_immune_exists and not agent.isZombie and roll < chance:
                agent.immune = True
                natural_immune_exists = True

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("move_phase")
        self.agents.shuffle_do("interact_phase")
        self.datacollector.collect(self)
        # The outbreak is over when no humans or no living zombies are left.
        if count_humans(self) == 0 or count_zombies(self) == 0:
            self.running = False


model_params = {
    "num_humans": {
        "type": "SliderInt",
        "value": 90,
        "label": "Starting humans:",
        "min": 0,
        "max": 300,
        "step": 5,
    },
    "num_zombies": {
        "type": "SliderInt",
        "value": 10,
        "label": "Starting zombies:",
        "min": 1,
        "max": 100,
        "step": 1,
    },
    "width": {
        "type": "SliderInt",
        "value": 20,
        "label": "Width:",
        "min": 10,
        "max": 50,
        "step": 5,
    },
    "height": {
        "type": "SliderInt",
        "value": 20,
        "label": "Height:",
        "min": 10,
        "max": 50,
        "step": 5,
    },
    "immunity": {
        "type": "Checkbox",
        "value": False,
        "label": "Immunity (1 in N starting humans is immune)",
    },
    "immunity_radius": {
        "type": "SliderInt",
        "value": 1,
        "label": "Immunity sharing radius:",
        "min": 1,
        "max": 3,
        "step": 1,
    },
    "shots_start": {
        "type": "SliderInt",
        "value": SHOTS_START,
        "label": "Starting shots per agent:",
        "min": 1,
        "max": 30,
        "step": 1,
    },
    "immune_can_die": {
        "type": "Checkbox",
        "value": False,
        "label": "Zombies kill immune humans with no ammo",
    },
    "seed": {"type": "InputText", "value": 42, "label": "Random seed (same seed, same run)"},
}


def agent_portrayal(agent):
    if agent.dead and agent.isZombie:
        return {"size": 15, "color": "tab:gray", "marker": "o"}
    if agent.dead:
        return {"size": 15, "color": "black", "marker": "o"}
    if agent.isZombie:
        return {"size": 40, "color": "tab:red", "marker": "o"}
    if agent.immune:
        return {"size": 40, "color": "tab:green", "marker": "o"}
    return {"size": 40, "color": "tab:blue", "marker": "o"}


model = OutbreakModel()

SpaceGraph = make_space_component(agent_portrayal)
CountPlot = make_plot_component(
    {
        "Humans": "tab:blue",
        "Immune": "tab:green",
        "Zombies": "tab:red",
        "DeadZombies": "tab:gray",
        "DeadHumans": "black",
    }
)

page = SolaraViz(
    model,
    components=[SpaceGraph, CountPlot],
    model_params=model_params,
    name="Zombie Outbreak",
)

# This is required to render the visualization in the Jupyter notebook
page


# Batch experiment script: python a1_abhai.py

CONFIGS = {
    f"{name}, {shots} shots": {"shots_start": shots, **settings}
    for shots in (15, 2) # Starting shots per human. More humans are killed by zombie when shots are lesser
    for name, settings in {
        "baseline": {"immunity": False, "immune_can_die": False},
        "immunity": {"immunity": True, "immune_can_die": False},
        "immunity+immune_can_die": {"immunity": True, "immune_can_die": True},
    }.items()
}
print(CONFIGS)
RUNS = (6, 8, 12, 13, 29)  # seeds that start with an immune human
MAX_STEPS = 200


def run_once(config, seed):
    m = OutbreakModel(seed=seed, **CONFIGS[config])
    while m.running and m.steps < MAX_STEPS:
        m.step()
    df = m.datacollector.get_model_vars_dataframe().reset_index(names="Step")
    df.insert(0, "seed", seed)
    df.insert(0, "config", config)
    return df


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    results = pd.concat(
        [run_once(c, s) for c in CONFIGS for s in RUNS], ignore_index=True
    )
    results.to_csv(os.path.join(here, "a1_results.csv"), index=False)
    print(f"wrote {len(results)} rows to a1_results.csv")
