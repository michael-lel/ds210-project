use std::collections::HashMap;
use ndarray::Array2;
use ndarray::s;

pub fn read_csv(file_name: &str) -> HashMap<(String, String), (Vec<String>, Vec<f64>)> {
    let mut rdr  = csv::Reader::from_path(file_name).unwrap();
    let mut data = HashMap::new();
    let headers = rdr.headers().unwrap();
    let stats_index = headers.iter().enumerate().find(|(_index, header)| *header == "Wins").map(|(index, _)| index).unwrap();
    for result in rdr.records() {
        let mut stats = Vec::new();
        let mut bio = Vec::new();
        match result {
            Ok(record) => {
                for i in 2..stats_index {
                    bio.push(record[i].to_string());
                }
                for i in stats_index..record.len() {
                    stats.push(record[i].parse::<f64>().unwrap());
                }
                data.insert((record[0].to_string(),record[1].to_string()), (bio, stats));
            },
            Err(err) => {
                println!("error reading CSV record {}", err);
            }  
        }
    }
    data
}

pub fn create_x(data: &HashMap<(String, String), (Vec<String>, Vec<f64>)>) -> Array2<f64> {
    let mut complete_stats = Vec::new();
    for ((_team, _year), (_bio, stats)) in data {
        complete_stats.extend(stats);
    }
    let num_teams = complete_stats.len()/18;
    let mut x_stats = Array2::from_shape_vec((num_teams, 18), complete_stats).unwrap();
    for i in 0..x_stats.shape()[1] {
        let column = x_stats.slice(s![..,i]);
        let col_mean = column.mean().unwrap();
        let col_stddev = column.std(1.0);
        for j in 0..x_stats.shape()[0] {
            let og_value = x_stats[[j,i]];
            x_stats[[j,i]] = (og_value - col_mean) / col_stddev;
        }
    }
    x_stats
}

pub fn create_y(data: &HashMap<(String, String), (Vec<String>, Vec<f64>)>) -> Array2<i8> {
    let mut tourny_wins = Vec::new();
    for ((_team, _year), (bio, _stats)) in data {
        match &bio[1] as &str {
            "NA" => tourny_wins.push(-1),
            "N/A" => tourny_wins.push(-1),
            "R68" => tourny_wins.push(0),
            "R64" => tourny_wins.push(0),
            "R32" => tourny_wins.push(1),
            "S16" => tourny_wins.push(2),
            "E8" => tourny_wins.push(3),
            "F4" => tourny_wins.push(4),
            "2ND" => tourny_wins.push(5),
            "Champions" => tourny_wins.push(6),
            &_ => println!("Improper data"),
        }
    }
    let num_teams = tourny_wins.len();
    let y_stats = Array2::from_shape_vec((num_teams, 1), tourny_wins).unwrap();
    y_stats
}
