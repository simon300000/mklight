//! Private straight-line double-float system preparation specification.
use crate::{Df32, FpEnv, Ftz};
use std::{
    cell::RefCell,
    collections::HashMap,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

pub const CONSTANT: u32 = 0;
pub const PARAMETER: u32 = 1;
pub const ADD: u32 = 2;
pub const SUB: u32 = 3;
pub const MUL: u32 = 4;
pub const DIV: u32 = 5;
pub const SQRT: u32 = 6;
pub const SELECT: u32 = 7; // a <= 0 ? b : c

#[derive(Default)]
pub(crate) struct Algebra {
    pub code: Vec<[u32; 4]>,
    constants: HashMap<u64, u32>,
    nodes: HashMap<[u32; 4], u32>,
    values: Vec<Option<f64>>,
}
pub(crate) type Builder = Rc<RefCell<Algebra>>;
#[derive(Clone)]
pub(crate) struct E(pub Builder, pub u32);
impl E {
    pub fn constant(b: &Builder, x: f64) -> Self {
        let mut a = b.borrow_mut();
        if let Some(&i) = a.constants.get(&x.to_bits()) {
            return Self(b.clone(), i);
        }
        let d = if x.is_finite() {
            Df32::from_f64(x)
        } else {
            Df32::new(x as f32, 0.0)
        };
        let i = a.code.len() as u32;
        a.code.push([CONSTANT, d.hi.to_bits(), d.lo.to_bits(), 0]);
        a.values.push(Some(x));
        a.constants.insert(x.to_bits(), i);
        Self(b.clone(), i)
    }
    pub fn parameter(b: &Builder, i: usize) -> Self {
        let mut a = b.borrow_mut();
        let id = a.code.len() as u32;
        a.code.push([PARAMETER, i as u32, 0, 0]);
        a.values.push(None);
        Self(b.clone(), id)
    }
    pub fn c(&self, x: f64) -> Self {
        Self::constant(&self.0, x)
    }
    pub fn value(&self) -> Option<f64> {
        self.0.borrow().values[self.1 as usize]
    }
    fn op(self, op: u32, b: Self, c: Self) -> Self {
        let folded = match (self.value(), b.value(), c.value()) {
            (Some(a), Some(b), Some(c)) => Some(match op {
                ADD => a + b,
                SUB => a - b,
                MUL => a * b,
                DIV => a / b,
                SQRT => a.sqrt(),
                SELECT => {
                    if a <= 0.0 {
                        b
                    } else {
                        c
                    }
                }
                _ => unreachable!(),
            }),
            _ => None,
        };
        if let Some(v) = folded {
            return self.c(v);
        }
        // Exact identities (up to the sign of a zero result).
        match (op, self.value(), b.value()) {
            (ADD, _, Some(0.0)) | (SUB, _, Some(0.0)) | (MUL | DIV, _, Some(1.0)) => return self,
            (ADD, Some(0.0), _) | (MUL, Some(1.0), _) => return b,
            _ => {}
        }
        // Identical operations share one node (common subexpressions).
        let node = [op, self.1, b.1, c.1];
        let mut a = self.0.borrow_mut();
        let id = match a.nodes.get(&node) {
            Some(&id) => id,
            None => {
                let id = a.code.len() as u32;
                a.code.push(node);
                a.values.push(None);
                a.nodes.insert(node, id);
                id
            }
        };
        drop(a);
        Self(self.0, id)
    }
    pub fn sqrt(self) -> Self {
        self.clone().op(SQRT, self.clone(), self)
    }
    pub fn select(self, nonpositive: Self, positive: Self) -> Self {
        self.op(SELECT, nonpositive, positive)
    }
    pub fn abs(self) -> Self {
        self.clone().select(-self.clone(), self)
    }
    /// Angles bounded to +/-360 degrees at the input boundary. Reduce by 16,
    /// evaluate sin/cos on [-pi/8,pi/8], then four exact doubling identities.
    /// No library transcendental or device-dependent range reduction.
    pub fn sin_cos(self) -> (Self, Self) {
        let x = self / 16.0;
        let xx = x.clone() * x.clone();
        // Horner with signed coefficients, through degree 15.
        let mut s = x.c(-1.0 / 1307674368000.0);
        for a in [
            1.0 / 6227020800.0,
            -1.0 / 39916800.0,
            1.0 / 362880.0,
            -1.0 / 5040.0,
            1.0 / 120.0,
            -1.0 / 6.0,
            1.0,
        ] {
            s = x.c(a) + xx.clone() * s;
        }
        s = x.clone() * s;
        let mut c = x.c(-1.0 / 87178291200.0);
        for a in [
            1.0 / 479001600.0,
            -1.0 / 3628800.0,
            1.0 / 40320.0,
            -1.0 / 720.0,
            1.0 / 24.0,
            -0.5,
            1.0,
        ] {
            c = x.c(a) + xx.clone() * c;
        }
        for _ in 0..4 {
            let sn = s.clone() * c.clone() * 2.0;
            c = c.clone() * c - s.clone() * s;
            s = sn;
        }
        (s, c)
    }
}
macro_rules! binary {
    ($trait:ident,$method:ident,$code:ident) => {
        impl $trait for E {
            type Output = Self;
            fn $method(self, b: Self) -> Self {
                let z = self.c(0.0);
                self.op($code, b, z)
            }
        }
        impl $trait<f64> for E {
            type Output = Self;
            fn $method(self, b: f64) -> Self {
                let b = self.c(b);
                self.$method(b)
            }
        }
    };
}
binary!(Add, add, ADD);
binary!(Sub, sub, SUB);
binary!(Mul, mul, MUL);
binary!(Div, div, DIV);
impl Neg for E {
    type Output = Self;
    fn neg(self) -> Self {
        self.c(0.0) - self
    }
}

pub(crate) fn evaluate(code: &[[u32; 4]], parameters: &[Df32], values: &mut [Df32]) {
    for i in 0..code.len() {
        let [op, a, b, c] = code[i];
        values[i] = match op {
            CONSTANT => Df32::new(f32::from_bits(a), f32::from_bits(b)),
            PARAMETER => parameters[a as usize],
            _ => {
                let (a, b, c) = (values[a as usize], values[b as usize], values[c as usize]);
                match op {
                    ADD => a.add_in::<Ftz>(b),
                    SUB => a.add_in::<Ftz>(-b),
                    MUL => a.mul_in::<Ftz>(b),
                    DIV if b.hi == 0.0 => Df32::new(Ftz::div(a.hi, b.hi), 0.0),
                    DIV => a.div_in::<Ftz>(b),
                    SQRT => a.sqrt_in::<Ftz>(),
                    SELECT => {
                        if a.hi < 0.0 || (a.hi == 0.0 && a.lo <= 0.0) {
                            b
                        } else {
                            c
                        }
                    }
                    _ => unreachable!("compiled system opcode"),
                }
            }
        };
    }
}
