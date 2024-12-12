mod read_csv;

fn main() {
    let cbb_2020_data = read_csv::read_csv("cbb_2020data.csv");
    let cbb_full_data = read_csv::read_csv("cbb_fulldata.csv");
    let x_train = read_csv::create_x(&cbb_full_data);
    let y_train = read_csv::create_y(&cbb_full_data);
    let x_test = read_csv::create_x(&cbb_2020_data);
    println!("{:?}", y_train.len());
}