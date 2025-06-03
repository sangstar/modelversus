pub struct DPMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    data: Vec<i16>,
}

impl DPMatrix {
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            data: vec![0; n_rows * n_cols],
            n_rows,
            n_cols,
        }
    }
    pub fn at(&self, i: usize, j: usize) -> i16 {
        self.data[i * self.n_cols + j]
    }
    pub fn set(&mut self, i: usize, j: usize, val: i16) {
        self.data[i * self.n_cols + j] = val;
    }
}

pub struct Sequence {
    pub text: String,
    pub word_vector: Vec<String>,
    pub n_words: usize,
}

impl Sequence {
    pub fn new(text: &str) -> Self {
        let word_vec = str_to_word_vec(&text);
        let word_vec_len = word_vec.len();
        Sequence {
            text: text.to_string(),
            word_vector: word_vec,
            n_words: word_vec_len,
        }
    }
}

pub fn str_to_word_vec(string: &str) -> Vec<String> {
    string.split_whitespace().map(|s| s.to_string()).collect()
}
