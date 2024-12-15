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

fn sort_hash(data: &HashMap<(String, String), (Vec<String>, Vec<f64>)>) -> Vec<&(String, String)> {
    let mut team_info = Vec::new();
    for (team_year, (_bio, _stats)) in data {
        team_info.push(team_year);
    }
    team_info.sort_by(|name, year| name.1.cmp(&year.1).then_with(|| name.0.cmp(&year.0)));
    team_info
}

pub fn create_x(data: &HashMap<(String, String), (Vec<String>, Vec<f64>)>) -> Array2<f64> {
    let mut complete_stats = Vec::new();
    for team_year in sort_hash(data) {
        let team_stats = &data.get(team_year).unwrap().1;
        complete_stats.extend(team_stats);
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

pub fn create_y(data: &HashMap<(String, String), (Vec<String>, Vec<f64>)>) -> Array2<f64> {
    let mut tourny_wins = Vec::new();
    for team_year in sort_hash(data) {
        let tourney_result = &data.get(team_year).unwrap().0[1];
        match &tourney_result as &str {
            "NA" => tourny_wins.push(0.0),
            "N/A" => tourny_wins.push(0.0),
            "R68" => tourny_wins.push(1.0),
            "R64" => tourny_wins.push(1.0),
            "R32" => tourny_wins.push(2.0),
            "S16" => tourny_wins.push(3.0),
            "E8" => tourny_wins.push(4.0),
            "F4" => tourny_wins.push(5.0),
            "2ND" => tourny_wins.push(6.0),
            "Champions" => tourny_wins.push(7.0),
            &_ => println!("Improper data"),
        }
    }
    let num_teams = tourny_wins.len();
    let y_stats = Array2::from_shape_vec((num_teams, 1), tourny_wins).unwrap();
    y_stats
}
