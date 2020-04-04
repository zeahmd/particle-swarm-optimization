import numpy as np

class Particle:
    """
    Particle class represents a solution inside a pool(Swarm).
    """
    def __init__(self, dim_shape, x_range, v_range):
        """
        Particle class constructor

        :param dim_shape: tuple(no_dim, )
            Shape of x(position), v(velocity).
        :param x_range: tuple(double)
            Min and Max value(range) of dimension.
        :param v_range: tuple(double)
            Min and Max value(range) of velocity.
        """
        self.x = np.random.uniform(x_range[0], x_range[1], dim_shape)
        self.v = np.random.uniform(v_range[0], v_range[1], dim_shape)
        self.pbest = np.inf
        self.pbestpos = np.zeros(dim_shape)


class Swarm:
    """
    Swarm class represents a pool of solution(particle).
    """
    def __init__(self, no_particle, dim_shape, x_range, v_range, iw_range, c):
        """
        Swarm class constructor.

        :param no_particle: int
            No of particles(solutions).
        :param dim_shape:  tuple(no_dim, )
            Shape of x(position), v(velocity).
        :param x_range: tuple(double)
            Min and Max value(range) of dimension.
        :param v_range: tuple(double)
            Min and Max value(range) of velocity.
        :param iw_range: tuple(double)
            Min and Max value(range) of interia weight.
        :param c: tuple(double)
            c[0] -> cognitive parameter, c[1] -> social parameter.
        """
        self.p = np.array([Particle(dim_shape, x_range, v_range) for i in range(no_particle)])
        self.gbest = np.inf
        self.gbestpos = np.zeros(dim_shape)
        self.x_range = x_range
        self.v_range = v_range
        self.c0 = c[0]
        self.c1 = c[1]
        self.iw_range = iw_range
        self.dim_shape = dim_shape
        self.update_particle_pos = None
        self.update_particle_vel = None

    def _update_particle_pos(self, p,  fitness):
        """
        It updates particle position.
        :param p: Particle
            Particle to updated position.
        :param fitness: double
            Fitness value or loss(to be optimized).
        :return: Particle
            Updated Particle.
        """
        if fitness < p.pbest:
            p.pbest = fitness
            p.pbestpos = p.x

        return p
            
    def _update_particle_vel(self, p):
        """
        It updates velocity of a particle.
        It is used by optimize function.
        :param p: Particle
            Particle to update velocity.
        :return: Particle
            Particle with updated velocity.
        """
        iw = np.random.uniform(self.iw_range[0], self.iw_range[1], self.dim_shape)
        p.v = iw * p.v + (self.c0 * np.random.uniform(0.0, 1.0, self.dim_shape) *\
        (p.pbestpos - p.x)) + (self.c1 * np.random.uniform(0.0, 1.0, self.dim_shape) * (self.gbestpos - p.x))
        p.v = p.v.clip(min=self.v_range[0], max=self.v_range[1])
        p.x = p.x + p.v
        p.x = p.x.clip(min=self.x_range[0], max=self.x_range[1])
        return p
        
    def optimize(self, function, print_step, iter):
        """
        optimize is used start optimization.

        :param function: function
            Function to be optimized.
        :param print_step: int
            Print pause between two adjacent prints.
        :param iter: int
            No of iterations.
        """
        function = np.vectorize(function)
        self.update_particle_pos = np.vectorize(self._update_particle_pos)
        self.update_particle_vel =  np.vectorize(self._update_particle_vel)


        for i in range(iter):
            fitness = function(self.p)
            self.p = self.update_particle_pos(self.p, fitness)
            min_fitness = fitness[np.argmin(fitness)]
            if min_fitness < self.gbest:
                self.gbest = min_fitness
                self.gbestpos = self.p[np.argmin(fitness)].x
                
            
            self.p = self.update_particle_vel(self.p)
            
            if i % print_step == 0:
                print("Iteration: ", i, " Loss/gbest: ", self.gbest, "Fitness: ", min_fitness)

        print("global best fitness: ", self.gbest)
            

def sphere(particle):
    """
    sphere is an objective function used to test
    optimization algorithms.

    :param particle: 1d Numpy Array of Particle
        List of position of particle in all dimensions.
    :return: double
        Calculated value of objective function.
    """
    _sum = 0.0
    for _x in particle.x:
        _sum = _sum + _x**2
    
    return _sum


if __name__ == "__main__":
    s = Swarm(20, (10, ), (-5.12, 5.12), (-2.0, 2.0), (0.4, 0.9), (1.49, 1.49))
    s.optimize(sphere, 100, 5000)
    print("Best Fitness/Min Loss: ", s.gbest)
    print("Best position/Best weights: ", s.gbestpos)