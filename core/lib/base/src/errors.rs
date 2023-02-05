#[derive(Debug)]
pub enum ANNError {
    GenericError { message: String },
    // QueryError, // Unknown { code: u8 },
    // Err41,
}
impl std::error::Error for ANNError {}

impl std::fmt::Display for ANNError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ANNError::GenericError { message } => write!(f, "generic error {}.", message),
            // ANNError::QueryError => write!(f, "Sit by a lake"),
        }
    }
}
