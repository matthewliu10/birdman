use std::iter::FromIterator;
use std::ops::Index;

#[derive(Clone, Debug)]
pub struct Chromosome {
    pub genes: Vec<f32>,
}

impl Chromosome {
    pub fn new(genes: Vec<f32>) -> Self {
        Chromosome { genes }
    }

    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn genes(&self) -> &Vec<f32> {
        &self.genes
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice(),)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chromosome() -> Chromosome {
        Chromosome {
            genes: vec![3.0, 1.0, 2.0],
        }
    }

    mod new {
        use super::*;

        #[test]
        fn test() {
            let genes = Chromosome::new(vec![3.0, 1.0, 2.0]).genes;
            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], 3.0);
            assert_eq!(genes[1], 1.0);
            assert_eq!(genes[2], 2.0);
        }
    }

    mod len {
        use super::*;

        #[test]
        fn test() {
            assert_eq!(chromosome().len(), 3);
        }
    }

    mod iter {
        use super::*;

        #[test]
        fn test() {
            let chromosome = chromosome();
            let genes: Vec<_> = chromosome.iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], &3.0);
            assert_eq!(genes[1], &1.0);
            assert_eq!(genes[2], &2.0);
        }
    }

    mod iter_mut {
        use super::*;

        #[test]
        fn test() {
            let mut chromosome = chromosome();
            chromosome.iter_mut().for_each(|gene| {
                *gene *= 10.0;
            });

            let genes: Vec<_> = chromosome.iter().collect();

            assert_eq!(genes.len(), 3);
            assert_eq!(genes[0], &30.0);
            assert_eq!(genes[1], &10.0);
            assert_eq!(genes[2], &20.0);
        }
    }

    mod index {
        use super::*;

        #[test]
        fn test() {
            let chromosome = chromosome();

            assert_eq!(chromosome[0], 3.0);
            assert_eq!(chromosome[1], 1.0);
            assert_eq!(chromosome[2], 2.0);
        }
    }

    mod from_iterator {
        use super::*;

        #[test]
        fn test() {
            let chromosome: Chromosome = vec![3.0, 1.0, 2.0].into_iter().collect();

            assert_eq!(chromosome[0], 3.0);
            assert_eq!(chromosome[1], 1.0);
            assert_eq!(chromosome[2], 2.0);
        }
    }
}
