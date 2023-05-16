//! provides a flatten() function that flattens any nested JSON object into
//! a key -> val relation, in a similar manner to what is done by elasticsearch
//!
//!

use serde_json::{json, Map, Value};

pub fn flatten(json: &Map<String, Value>) -> Map<String, Value> {
    let mut obj = Map::new();
    insert_object(&mut obj, None, json);
    obj
}

fn insert_object(
    base_json: &mut Map<String, Value>,
    base_key: Option<&str>,
    object: &Map<String, Value>,
) {
    for (key, value) in object {
        let new_key = base_key.map_or_else(|| key.clone(), |base_key| format!("{base_key}.{key}"));

        if let Some(array) = value.as_array() {
            insert_array(base_json, &new_key, array);
        } else if let Some(object) = value.as_object() {
            insert_object(base_json, Some(&new_key), object);
        } else {
            insert_value(base_json, &new_key, value.clone());
        }
    }
}

fn insert_array(base_json: &mut Map<String, Value>, base_key: &str, array: &Vec<Value>) {
    for value in array {
        if let Some(object) = value.as_object() {
            insert_object(base_json, Some(base_key), object);
        } else if let Some(sub_array) = value.as_array() {
            insert_array(base_json, base_key, sub_array);
        } else {
            insert_value(base_json, base_key, value.clone());
        }
    }
}

fn insert_value(base_json: &mut Map<String, Value>, key: &str, to_insert: Value) {
    debug_assert!(!to_insert.is_object());
    debug_assert!(!to_insert.is_array());

    // does the field aleardy exists?
    if let Some(value) = base_json.get_mut(key) {
        // is it already an array
        if let Some(array) = value.as_array_mut() {
            array.push(to_insert);
        // or is there a collision
        } else {
            let value = std::mem::take(value);
            base_json[key] = json!([value, to_insert]);
        }
        // if it does not exist we can push the value untouched
    } else {
        base_json.insert(key.to_string(), json!(to_insert));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_obj_simple() {
        let mut base: Value = json!({
          "a": {
            "b": "c",
            "d": "e",
            "f": "g"
          }
        });
        let json = std::mem::take(base.as_object_mut().unwrap());
        let flat = flatten(&json);

        assert_eq!(
            &flat,
            json!({
                "a.b": "c",
                "a.d": "e",
                "a.f": "g"
            })
            .as_object()
            .unwrap()
        );
    }
    #[test]
    fn flatten_obj_with_arrs() {
        let mut base: Value = json!({
          "title": "Population dynamics of head-direction neurons during drift and reorientation",
          "paragraphs": [
            {
                "content": "In the version of this article initially published, there was a typographical error in the seventh sentence",
                "section": "CORRECTION",
            },
            {
                "content": "of the Methods “Surgeries” subsection, where in the text now reading",
                "section": "CORRECTION",
            },
            {
                "content": "where in the text now reading \"One week after injection, a 0.5-mm-diameter gradient refractive index (GRIN)",
                "section": "CORRECTION",
            },
          ],
        });
        let json = std::mem::take(base.as_object_mut().unwrap());
        let flat = flatten(&json);
        assert_eq!(
            &flat,
            json!({
                "paragraphs.content": [""],
                "paragraphs.section": ["CORRECTION", "CORRECTION", "CORRECTION"],
                // "paragraphs.": "g"
            })
            .as_object()
            .unwrap()
        );
    }
}
