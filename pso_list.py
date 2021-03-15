import numpy as np


class Particle:
    """
    Particle class represents a solution inside a pool(Swarm).
    """

    def __init__(self, no_dim, x_range, v_range, fitness):
        """
        Particle class constructor

        :param no_dim: int
            No of dimensions.
        :param x_range: tuple(double)
            Min and Max value(range) of dimension.
        :param v_range: tuple(double)
            Min and Max value(range) of velocity.
        :param fitness: double
            fitness value is assigned to pbest initially.
        """
        self.x = [
            np.random.uniform(x_range[0], x_range[1], 1)[0] for i in range(no_dim)
        ]
        self.v = [
            np.random.uniform(v_range[0], v_range[1], 1)[0] for i in range(no_dim)
        ]
        self.pbest = float("inf")
        self.pbestpos = [0 for i in range(no_dim)]


class Swarm:
    """
    Swarm class represents a pool of solution(particle).
    """

    def __init__(self, no_particle, no_dim, x_range, v_range, iw_range, c, fitness):
        """
        Swarm class constructor.

        :param no_particle: int
            No of particles(solutions).
        :param no_dim: int
            No of dimensions.
        :param x_range: tuple(double)
            Min and Max value(range) of dimension.
        :param v_range: tuple(double)
            Min and Max value(range) of velocity.
        :param iw_range: tuple(double)
            Min and Max value(range) of interia weight.
        :param c: tuple(double)
            c[0] -> cognitive parameter, c[1] -> social parameter.
        :param fitness: double
            fitness value value assigned to pbest initially.
        """
        self.p = [
            Particle(no_dim, x_range, v_range, fitness) for i in range(no_particle)
        ]
        self.gbest = float("inf")
        self.gbestpos = [0 for i in range(no_dim)]
        self.no_dim = no_dim
        self.x_range = x_range
        self.v_range = v_range
        self.iw_range = iw_range
        self.c0 = c[0]
        self.c1 = c[1]

    def optimize(self, function, max_iter):
        """
        optimize is used start optimization.

        :param function: function
            Function to be optimized.
        :param max_iter: int
            No of iterations.
        """
        fitness = None
        for i in range(max_iter):
            for particle in self.p:
                fitness = function(particle.x)

                if fitness < particle.pbest:
                    particle.pbest = fitness
                    for d in range(self.no_dim):
                        particle.pbestpos[d] = particle.x[d]

                if fitness < self.gbest:
                    self.gbest = fitness
                    for d in range(self.no_dim):
                        self.gbestpos[d] = particle.x[d]

            for particle in self.p:
                for d in range(self.no_dim):
                    iw = np.random.uniform(self.iw_range[0], self.iw_range[1], 1)[0]
                    particle.v[d] = (
                        iw * particle.v[d]
                        + (
                            self.c0
                            * np.random.uniform(0.0, 1.0, 1)[0]
                            * (particle.pbestpos[d] - particle.x[d])
                        )
                        + (
                            self.c1
                            * np.random.uniform(0.0, 1.0, 1)[0]
                            * (self.gbestpos[d] - particle.x[d])
                        )
                    )
                    if particle.v[d] > self.v_range[1]:
                        particle.v[d] = self.v_range[1]
                    if particle.v[d] < self.v_range[0]:
                        particle.v[d] = self.v_range[0]

                    particle.x[d] = particle.x[d] + particle.v[d]

                    if particle.x[d] > self.x_range[1]:
                        particle.x[d] = self.x_range[1]
                    if particle.x[d] < self.x_range[0]:
                        particle.x[d] = self.x_range[0]

            if i % 100 == 0:
                print("iteration#: ", i, " fitness: ", fitness, " gbest: ", self.gbest)


def sphere(particle_pos):
    """

    sphere is an objective function used to test
    optimization algorithms.

    :param particle_pos: 1d Numpy Array of Particle
        List of position of particle in all dimensions.
    :return: double
        Calculated value of objective function.
    """
    _sum = 0.0
    for d in particle_pos:
        _sum = _sum + d ** 2

    return _sum


if __name__ == "__main__":
    s = Swarm(20, 10, (-5.12, 5.12), (-2.0, 2.0), (0.4, 0.9), (1.49, 1.49), 100000.0)
    s.optimize(sphere, 5000)
    print("Printing best solution...")
    print(s.gbestpos)
