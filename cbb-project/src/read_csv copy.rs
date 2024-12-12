use std::collections::HashMap;

pub fn read_csv(file_name: &str) -> HashMap<String, (Vec<String>, Vec<f64>)>{
    let mut rdr  = csv::Reader::from_path(file_name).unwrap();
    let mut data = HashMap::new();
    let headers = rdr.headers().unwrap();
    let stats_index = headers.iter().enumerate().find(|(_index, header)| *header == "Games Played").map(|(index, _)| index).unwrap();
    for result in rdr.records() {
        let mut stats = Vec::new();
        let mut bio = Vec::new();
        match result {
            Ok(record) => {
                for i in 1..stats_index {
                    bio.push(record[i].to_string());
                }
                for i in stats_index..record.len() {
                    stats.push(record[i].parse::<f64>().unwrap());
                }
                data.insert(record[0].to_string(), (bio, stats));
            },
            Err(err) => {
                println!("error reading CSV record {}", err);
            }  
        }
    }
    data
}
