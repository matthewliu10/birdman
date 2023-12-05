pub use self::{animal::*, animal_individual::*, brain::*, eye::*, food::*, world::*};

mod animal;
mod animal_individual;
mod brain;
mod eye;
mod food;
mod world;

use lib_genetic_algorithm as ga;
use nalgebra as na;
use rand::{Rng, RngCore};
use std::f32::consts::FRAC_PI_2;

const MIN_SPEED:f32 = 0.001;
const MAX_SPEED:f32 = 0.003;
const SPEED_ACCEL: f32 = 0.02;
const ROTATION_ACCEL: f32 = FRAC_PI_2;
const GENERATION_LENGTH: usize = 2500;

pub struct Simulation {
    world: World,
    ga: ga::GeneticAlgorithm<ga::RouletteWheelSelection,
    ga::UniformCrossover,
    ga::GaussianMutation>,
    age: usize,
}

impl Simulation {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            world: World::random(rng),
            ga: ga::GeneticAlgorithm::new(
                ga::RouletteWheelSelection::new(),
                ga::UniformCrossover::new(),
                ga::GaussianMutation::new(0.01, 0.3),
            ),
            age: 0,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn step(&mut self, rng: &mut dyn RngCore) -> Option<ga::Statistics> {
        self.process_collisions(rng);
        self.process_brains();
        self.process_movement();

        self.age += 1;

        if self.age > GENERATION_LENGTH {
            self.age = 0;
            Some(self.evolve(rng))
        } else {
            None
        }
    }

    fn process_collisions(&mut self, rng: &mut dyn RngCore) {
        for animal in &mut self.world.animals {
            for food in &mut self.world.foods {
                let distance = na::distance(&animal.position, &food.position);

                if distance <= 0.01 {
                    food.position = rng.gen();
                    animal.food_eaten += 1;
                }
            }
        }
    }

    fn process_brains(&mut self) {
        for animal in &mut self.world.animals {
            let vision = animal.eye.process_vision(
                animal.position,
                animal.rotation,
                &self.world.foods,
            );
            
            let adjustment = animal.brain.propogate(vision);

            let delta_speed = adjustment[0].clamp(-SPEED_ACCEL, SPEED_ACCEL);
            let delta_rotation = adjustment[1].clamp(-ROTATION_ACCEL, ROTATION_ACCEL);

            animal.speed = (animal.speed + delta_speed).clamp(MIN_SPEED, MAX_SPEED);
            animal.rotation = na::Rotation2::new(animal.rotation.angle() + delta_rotation);
        }
    }

    fn process_movement(&mut self) {
        for animal in &mut self.world.animals {
            animal.position += animal.rotation * na::Vector2::new(0.0, animal.speed);

            animal.position.x = na::wrap(animal.position.x, 0.0, 1.0);
            animal.position.y = na::wrap(animal.position.y, 0.0, 1.0);
        }
    }

    pub fn train(&mut self, rng: &mut dyn RngCore) -> ga::Statistics {
        loop {
            if let Some(stats) = self.step(rng) {
                return stats;
            }
        }
    }

    fn evolve(&mut self, rng: &mut dyn RngCore) -> ga::Statistics{
        let animal_individuals: Vec<_> = self.world.animals.iter().map( AnimalIndividual::from_animal).collect();

        let (new_population, stats) = self.ga.evolve(rng, &animal_individuals);

        self.world.animals = new_population.into_iter().map(|individual| individual.to_animal(rng)).collect();

        for food in &mut self.world.foods {
            food.position = rng.gen();
        }

        stats
    }
}
