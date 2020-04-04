#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <limits>
using namespace std;

#define P 20
#define DIM 10

class Particle
{
public:
    double *x, *v, pbest, *pbestpos;

    Particle()
    {
    }

    Particle(int no_dim)
    {
        x = new double[no_dim];
        v = new double[no_dim];
        pbest = numeric_limits<double>::infinity();
        pbestpos = new double[no_dim];
    }

    void intialize(int no_dim, double xmin, double xmax, double vmin, double vmax)
    {
        x = new double[no_dim];
        v = new double[no_dim];
        pbest = numeric_limits<double>::infinity();
        pbestpos = new double[no_dim];

        for (int i = 0; i < no_dim; i++)
        {
            x[i] = xmin + double((xmax - xmin) * rand() / (RAND_MAX + 1.0));
            v[i] = vmin + double((vmax - vmin) * rand() / (RAND_MAX + 1.0));
        }
    }
};

double sphere(Particle p)
{
    double sum = 0.0;
    for (int i = 0; i < DIM; i++)
    {
        sum = sum + p.x[i] * p.x[i];
    }
    return sum;
}

class Swarm
{
public:
    Particle *p;
    double gbest, *gbestpos, xmin, xmax, vmin, vmax, iw, rmin, rmax, iwmin, iwmax, c0, c1;

    Swarm(int no_particle, int no_dim, double xmin, double xmax,
          double vmin, double vmax, double rmin, double rmax,
          double iwmin, double iwmax)
    {
        this->gbest = numeric_limits<double>::infinity();
        this->gbestpos = new double[no_dim];
        this->xmin = xmin;
        this->xmax = xmax;
        this->vmin = vmin;
        this->vmax = vmax;
        this->rmin = rmin;
        this->rmax = rmax;
        this->iwmin = iwmin;
        this->iwmax = iwmax;
        this->c0 = 1.49;
        this->c1 = 1.49;
        this->p = new Particle[no_particle];
        for (int i = 0; i < no_particle; i++)
        {
            this->p[i].intialize(no_dim, this->xmin, this->xmax, this->vmin, this->vmax);
        }
    }

    void optimize(int print_step, int iter)
    {
        double tfitness = 0.0;
        for (int i = 1; i <= iter; i++)
        {
            for (int j = 0; j < P; j++)
            {
                tfitness = sphere(this->p[j]);

                if (tfitness < this->p[j].pbest)
                {
                    this->p[j].pbest = tfitness;
                    for (int k = 0; k < DIM; k++)
                    {
                        this->p[j].pbestpos[k] = this->p[j].x[k];
                    }
                }

                if (tfitness < this->gbest)
                {
                    gbest = tfitness;
                    for (int k = 0; k < DIM; k++)
                    {
                        this->gbestpos[k] = this->p[j].x[k];
                    }
                }
            }

            for (int j = 0; j < P; j++)
            {
                for (int k = 0; k < DIM; k++)
                {
                    double rand1 = rmin + double((rmax - rmin) * rand() / (RAND_MAX + 1.0));
                    double rand2 = rmin + double((rmax - rmin) * rand() / (RAND_MAX + 1.0));
                    iw = iwmin + double((iwmax - iwmin) * rand() / (RAND_MAX + 1.0));

                    this->p[j].v[k] = iw * this->p[j].v[k] + c0 * rand1 * (this->p[j].pbestpos[k] - this->p[j].x[k]) + c1 * rand2 * (this->gbestpos[k] - this->p[j].x[k]);

                    if (this->p[j].v[k] > vmax)
                        this->p[j].v[k] = vmax;
                    if (this->p[j].v[k] < vmin)
                        this->p[j].v[k] = vmin;

                    this->p[j].x[k] = (this->p[j].x[k] + this->p[j].v[k]);

                    if (this->p[j].x[k] > xmax)
                        this->p[j].x[k] = xmax;
                    if (this->p[j].x[k] < xmin)
                        this->p[j].x[k] = xmin;
                }
            }

            if (i % print_step == 0)
            {
                cout << "Iteration#: " << i << "\t|| current iteration fitness: " << tfitness << "\t\t|| gbest(global fitness): " << this->gbest << endl;
            }
        }
    }
};

int main()
{
    Swarm *s = new Swarm(20, 10, -5.12, 5.12, -2.0, 2.0, 0.0, 1.0, 0.4, 0.9);
    s->optimize(100, 5000);

    getchar();
}