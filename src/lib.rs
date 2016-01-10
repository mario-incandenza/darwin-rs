extern crate time;
extern crate simple_parallel;
extern crate rand;

// external modules
use time::precise_time_ns;
use simple_parallel::Pool;
use rand::Rng;

#[derive(Debug,Clone)]
pub enum SimulationType {
    EndIteration(u32),
    EndFittness(f64),
    EndFactor(f64)
}

#[derive(Debug,Clone)]
pub enum FittestType {
    GlobalFittest,
    LocalFittest,
    RandomFittest
}

pub struct Simulation<T: 'static + Individual + Send> {
    pub type_of_simulation: SimulationType,
    pub num_of_individuals: u32,
    pub num_of_threads: usize,
    pub improvement_factor: f64,
    pub original_fittness: f64,
    pub population: Vec<IndividualWrapper<T>>,
    pub total_time_in_ms: f64,
    pub iteration_counter: u32,
    pub output_new_fittest: bool,
    pub type_of_fittest: FittestType,
    pub run_body: Box<Fn(&mut Simulation<T>, IndividualWrapper<T>, &mut Pool) -> IndividualWrapper<T>>
}

fn find_fittest<T: Individual + Clone + Send>(simulation: &mut Simulation<T>, fittest: IndividualWrapper<T>) -> IndividualWrapper<T> {
    let mut fittest = fittest;

    for wrapper in simulation.population.iter() {
        if wrapper.fittness < fittest.fittness {
            fittest = wrapper.clone();
            if simulation.output_new_fittest {
                println!("new fittest: {}", fittest.fittness);
            }
        }
    }

    fittest
}

fn run_body_global_fittest<T: Individual + Clone + Send>(simulation: &mut Simulation<T>,
    global_fittest: IndividualWrapper<T>, pool: &mut Pool) -> IndividualWrapper<T> {
    let mut fittest = global_fittest;

    pool.for_(simulation.population.iter_mut(), |wrapper|
        {
            for _ in 0..wrapper.num_of_mutations {
                wrapper.individual.mutate();
            }
            wrapper.fittness = wrapper.individual.calculate_fittness();
        }
    );

    // Find fittest individual for whole simulation...
    fittest = find_fittest(simulation, fittest);

    // ...  and copy it to the others (except the last one, to avoid local minimum or maximum)
    for i in 0..(simulation.population.len() - 1) {
        simulation.population[i].individual = fittest.individual.clone();
    }

    // Set fittness of first individual, since population vector will be sorted (by fittness) after the loop
    simulation.population[0].fittness = fittest.fittness;

    fittest
}

fn run_body_local_fittest<T: Individual + Clone + Send>(simulation: &mut Simulation<T>,
    global_fittest: IndividualWrapper<T>, pool: &mut Pool) -> IndividualWrapper<T> {
    let mut fittest = simulation.population[0].clone();

    pool.for_(simulation.population.iter_mut(), |wrapper|
        {
            for _ in 0..wrapper.num_of_mutations {
                wrapper.individual.mutate();
            }
            wrapper.fittness = wrapper.individual.calculate_fittness();
        }
    );

    // Find fittest individual only for this function call...
    fittest = find_fittest(simulation, fittest);

    simulation.improvement_factor = fittest.fittness / simulation.original_fittness;

    // ...  and copy it to the others (except the last one, to avoid local minimum or maximum)
    for i in 0..(simulation.population.len() - 1) {
        simulation.population[i].individual = fittest.individual.clone();
    }

    // Set fittness of first individual, since population vector will be sorted (by fittness) after the loop
    simulation.population[0].fittness = fittest.fittness;

    fittest
}

fn run_body_random_fittest<T: Individual + Clone + Send>(simulation: &mut Simulation<T>,
    global_fittest: IndividualWrapper<T>, pool: &mut Pool) -> IndividualWrapper<T> {
    let mut fittest = global_fittest;

    pool.for_(simulation.population.iter_mut(), |wrapper|
        {
            for _ in 0..wrapper.num_of_mutations {
                wrapper.individual.mutate();
            }
            wrapper.fittness = wrapper.individual.calculate_fittness();
        }
    );

    // Find fittest individual for whole simulation...
    fittest = find_fittest(simulation, fittest);

    simulation.improvement_factor = fittest.fittness / simulation.original_fittness;

    // ... and choose one random individual to set it back to the fittest
    let mut rng = rand::thread_rng();

    let index: usize = rng.gen_range(0, simulation.population.len());

    simulation.population[index].individual = fittest.individual.clone();

    fittest
}

impl<T: Individual + Clone + Send> Simulation<T> {
    pub fn run(&mut self) {
        let start_time = precise_time_ns();

        self.original_fittness = self.population[0].individual.calculate_fittness();

        // Initialize
        let mut fittest = self.population[0].clone();
        let mut iteration_counter = 0;
        let mut pool = simple_parallel::Pool::new(self.num_of_threads);

        match self.type_of_simulation {
            SimulationType::EndIteration(end_iteration) => {
                match self.type_of_fittest {
                    FittestType::GlobalFittest => {
                        for _ in 0..end_iteration {
                            fittest = run_body_global_fittest(self, fittest, &mut pool);
                        }
                    },
                    FittestType::LocalFittest => {
                        for _ in 0..end_iteration {
                            fittest = run_body_local_fittest(self, fittest, &mut pool);
                        }
                    },
                    FittestType::RandomFittest => {
                        for _ in 0..end_iteration {
                            fittest = run_body_random_fittest(self, fittest, &mut pool);
                        }
                    }
                }

                iteration_counter = end_iteration;
            },
            SimulationType::EndFactor(end_factor) => {
                match self.type_of_fittest {
                    FittestType::GlobalFittest => {
                        loop {
                            if self.improvement_factor <= end_factor { break }
                            fittest = run_body_global_fittest (self, fittest, &mut pool);
                            iteration_counter = iteration_counter + 1;
                        }
                    },
                    FittestType::LocalFittest => {
                        loop {
                            if self.improvement_factor <= end_factor { break }
                            fittest = run_body_local_fittest(self, fittest, &mut pool);
                            iteration_counter = iteration_counter + 1;
                        }
                    },
                    FittestType::RandomFittest => {
                        loop {
                            if self.improvement_factor <= end_factor { break }
                            fittest = run_body_random_fittest (self, fittest, &mut pool);
                            iteration_counter = iteration_counter + 1;
                        }
                    }
                }
            },
            SimulationType::EndFittness(end_fittness) => {
                match self.type_of_fittest {
                    FittestType::GlobalFittest => {
                        loop {
                            if fittest.fittness <= end_fittness { break }
                            fittest = run_body_global_fittest(self, fittest, &mut pool);
                            iteration_counter = iteration_counter + 1;
                        }
                    },
                    FittestType::LocalFittest => {
                        loop {
                            if self.population[0].fittness <= end_fittness { break }
                            fittest = run_body_local_fittest(self, fittest, &mut pool);
                            iteration_counter = iteration_counter + 1;
                        }
                    },
                    FittestType::RandomFittest => {
                        loop {
                            if fittest.fittness <= end_fittness { break }
                            fittest = run_body_random_fittest(self, fittest, &mut pool);
                            iteration_counter = iteration_counter + 1;
                        }
                    }
                }
            }
        }

        // sort all individuals by fittness
        self.population.sort_by(|a, b| a.fittness.partial_cmp(&b.fittness).unwrap());

        let end_time = precise_time_ns();

        self.total_time_in_ms = ((end_time - start_time) as f64) / (1000.0 * 1000.0);
        self.iteration_counter = iteration_counter;
    }

    pub fn print_fittness(&self) {
        for wrapper in self.population.iter() {
            println!("fittness: {}", wrapper.fittness);
        }
    }
}

#[derive(Debug,Clone)]
pub struct IndividualWrapper<T: Individual> {
    pub individual: T,
    fittness: f64,
    num_of_mutations: u32
}

pub trait Individual {
    fn mutate(&mut self);
    fn calculate_fittness(&self) -> f64;
}

pub struct SimulationBuilder<T: 'static + Individual + Send> {
    simulation: Simulation<T>
}

pub enum BuilderResult<T: 'static + Individual + Send> {
        LowIterration,
        LowIndividuals,
        Ok(Simulation<T>)
}

impl<T: Individual + Clone + Send> SimulationBuilder<T> {
    pub fn new() -> SimulationBuilder<T> {
        SimulationBuilder {
            simulation: Simulation {
                type_of_simulation: SimulationType::EndIteration(10),
                num_of_individuals: 10,
                num_of_threads: 2,
                improvement_factor: std::f64::MAX,
                original_fittness: std::f64::MAX,
                population: Vec::new(),
                total_time_in_ms: 0.0,
                iteration_counter: 0,
                output_new_fittest: true,
                run_body: Box::new(run_body_global_fittest),
                type_of_fittest: FittestType::GlobalFittest
            }
        }
    }

    pub fn iterations(mut self, iterations: u32) -> SimulationBuilder<T> {
        self.simulation.type_of_simulation = SimulationType::EndIteration(iterations);
        self
    }

    pub fn factor(mut self, factor: f64) -> SimulationBuilder<T> {
        self.simulation.type_of_simulation = SimulationType::EndFactor(factor);
        self
    }

    pub fn fittness(mut self, fittness: f64) -> SimulationBuilder<T> {
        self.simulation.type_of_simulation = SimulationType::EndFittness(fittness);
        self
    }

    pub fn individuals(mut self, individuals: u32) -> SimulationBuilder<T> {
        self.simulation.num_of_individuals = individuals;
        self
    }

    pub fn threads(mut self, threads: usize) -> SimulationBuilder<T> {
        self.simulation.num_of_threads = threads;
        self
    }

    pub fn output_new_fittest(mut self, output_new_fittest: bool) -> SimulationBuilder<T> {
        self.simulation.output_new_fittest = output_new_fittest;
        self
    }

    pub fn global_fittest(mut self) -> SimulationBuilder<T> {
        self.simulation.type_of_fittest = FittestType::GlobalFittest;
        self.simulation.run_body = Box::new(run_body_global_fittest);
        self
    }

    pub fn local_fittest(mut self) -> SimulationBuilder<T> {
        self.simulation.type_of_fittest = FittestType::LocalFittest;
        self.simulation.run_body = Box::new(run_body_local_fittest);
        self
    }

    pub fn random_fittest(mut self) -> SimulationBuilder<T> {
        self.simulation.type_of_fittest = FittestType::RandomFittest;
        self.simulation.run_body = Box::new(run_body_random_fittest);
        self
    }

    pub fn initial_population(mut self, initial_population: Vec<T>) -> SimulationBuilder<T>  {
        let mut new_population = Vec::new();

        for individual in initial_population {
            new_population.push(
                IndividualWrapper {
                    individual: individual,
                    fittness: std::f64::MAX,
                    num_of_mutations: 1,
                }
            )
        }

        let num_of_individuals = new_population.len() as u32;
        self.simulation.population = new_population;
        self.simulation.num_of_individuals = num_of_individuals;
        self
    }

    pub fn initial_population_num_mut(mut self, initial_population: Vec<(T, u32)>) -> SimulationBuilder<T>  {
        let mut new_population = Vec::new();

        for (individual, num_of_mutation) in initial_population {
            new_population.push(
                IndividualWrapper {
                    individual: individual,
                    fittness: std::f64::MAX,
                    num_of_mutations: num_of_mutation,
                }
            )
        }

        let num_of_individuals = new_population.len() as u32;
        self.simulation.population = new_population;
        self.simulation.num_of_individuals = num_of_individuals;
        self
    }

    pub fn one_individual(mut self, individual: T) -> SimulationBuilder<T> {
        for _ in 0..self.simulation.num_of_individuals {
            self.simulation.population.push(
                IndividualWrapper {
                    individual: individual.clone(),
                    fittness: std::f64::MAX,
                    num_of_mutations: 1,
                }
            );
        }
        self
    }

    pub fn one_individual_num_mut(mut self, individual: T, num_of_mutations: u32) -> SimulationBuilder<T> {
        for _ in 0..self.simulation.num_of_individuals {
            self.simulation.population.push(
                IndividualWrapper {
                    individual: individual.clone(),
                    fittness: std::f64::MAX,
                    num_of_mutations: num_of_mutations,
                }
            );
        }
        self
    }

    pub fn increasing_mutation_rate(mut self) -> SimulationBuilder<T> {
        let mut mutation_rate = 1;

        for wrapper in self.simulation.population.iter_mut() {
            wrapper.num_of_mutations = mutation_rate;
            mutation_rate = mutation_rate + 1;
        }

        self
    }

    pub fn finalize(self) -> BuilderResult<T> {
        let result = Simulation {
            type_of_simulation: self.simulation.type_of_simulation.clone(),
            num_of_individuals: self.simulation.num_of_individuals,
            num_of_threads: self.simulation.num_of_threads,
            improvement_factor: self.simulation.improvement_factor,
            original_fittness: self.simulation.original_fittness,
            population: self.simulation.population,
            total_time_in_ms: self.simulation.total_time_in_ms,
            iteration_counter: self.simulation.iteration_counter,
            output_new_fittest: self.simulation.output_new_fittest,
            run_body: self.simulation.run_body,
            type_of_fittest: self.simulation.type_of_fittest
        };

        if self.simulation.num_of_individuals < 3 { return BuilderResult::LowIndividuals }

        if let SimulationType::EndIteration(end_iteration) = self.simulation.type_of_simulation {
            if end_iteration < 10 { return BuilderResult::LowIterration }
        }

        BuilderResult::Ok(result)
    }
}
