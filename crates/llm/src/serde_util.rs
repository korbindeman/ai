//! Serde helpers.

use serde::{Deserialize, Deserializer, Serializer};
use std::time::Duration;

pub fn serialize_opt_duration_millis<S>(
    value: &Option<Duration>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(duration) => {
            serializer.serialize_some(&u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        }
        None => serializer.serialize_none(),
    }
}

pub fn deserialize_opt_duration_millis<'de, D>(
    deserializer: D,
) -> Result<Option<Duration>, D::Error>
where
    D: Deserializer<'de>,
{
    let millis: Option<u64> = Option::deserialize(deserializer)?;
    Ok(millis.map(Duration::from_millis))
}
