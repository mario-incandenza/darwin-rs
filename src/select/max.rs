// file: max.rs
//
// Copyright 2015-2017 The RsGenetic Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use Individual;
use super::*;

/// Selects best performing phenotypes from the population.
#[derive(Clone, Copy, Debug)]
pub struct MaximizeSelector {
    count: usize,
}

impl MaximizeSelector {
    /// Create and return a maximizing selector.
    ///
    /// Such a selector selects only the `count` best performing phenotypes
    /// as parents.
    ///
    /// * `count`: must be larger than zero, a multiple of two and less than the population size.
    pub fn new(count: usize) -> MaximizeSelector {
        MaximizeSelector { count: count }
    }
}

impl<I> Selector<I> for MaximizeSelector
where
    I: Individual + Clone + Send,
{
    fn select(&self, population: &[I]) -> Result<Parents<I>, ()> {
        if self.count == 0 || self.count % 2 != 0 || self.count * 2 >= population.len() {
            panic!(format!(
                "Invalid parameter `count`: {}. Should be larger than zero, a \
                            multiple of two and less than half the population size.",
                self.count
            ));
        }

        let mut scored = Vec::new();
        for _ind in population {
            let mut ind = _ind.clone();
            let score = ind.calculate_fitness();
            scored.push((score, ind));
        }

        scored.sort_by(|ref x, ref y| {
            y.0.partial_cmp(&x.0).unwrap_or(Ordering::Less)
        });

        let trunc: Vec<I> = scored
            .into_iter()
            .take(self.count)
            .map(|(_, ind)| ind)
            .collect();

        let mut index = 0;
        let mut result: Parents<I> = Vec::new();
        while index < trunc.len() {
            result.push((trunc[index].clone(), trunc[index + 1].clone()));
            index += 2;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;
    use select::*;
    use test::Test;

    #[test]
    fn test_count_zero() {
        let selector = MaximizeSelector::new(0);
        let population: Vec<Test> = (0..100).map(|i: usize| Test { f: i as f64 }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_count_odd() {
        let selector = MaximizeSelector::new(5);
        let population: Vec<Test> = (0..100).map(|i: usize| Test { f: i as f64 }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_count_too_large() {
        let selector = MaximizeSelector::new(100);
        let population: Vec<Test> = (0..100).map(|i: usize| Test { f: i as f64 }).collect();
        assert!(selector.select(&population).is_err());
    }

    #[test]
    fn test_result_size() {
        let selector = MaximizeSelector::new(20);
        let population: Vec<Test> = (0..100).map(|i: usize| Test { f: i as f64 }).collect();
        assert_eq!(20, selector.select(&population).unwrap().len() * 2);
    }

    #[test]
    fn test_result_ok() {
        let selector = MaximizeSelector::new(20);
        let population: Vec<Test> = (0..100).map(|i: usize| Test { f: i as f64 }).collect();
        // The greatest fitness should be 99.
        assert!(
            selector.select(&population).unwrap()[0]
                .0
                .calculate_fitness() == 99.0
        );
    }

    #[test]
    fn test_contains_best() {
        let selector = MaximizeSelector::new(2);
        let population: Vec<Test> = (0..100).map(|i: usize| Test { f: i as f64 }).collect();
        let mut parents = selector.select(&population).unwrap()[0];
        let fit1 = parents.0.calculate_fitness();
        let mut all_fitness = Vec::new();
        for i in population {
            all_fitness.push(OrderedFloat(i.clone().calculate_fitness()));
        }
        let max_fitness = all_fitness.iter().max().unwrap();

        assert_eq!(fit1, max_fitness.into_inner());
    }
}
