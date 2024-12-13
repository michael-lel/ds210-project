mod data_setup;

fn main() {
    let cbb_2020_data = data_setup::read_csv("cbb_2020data.csv");
    let cbb_full_data = data_setup::read_csv("cbb_fulldata.csv");
    let x_train = data_setup::create_x(&cbb_full_data);
    let y_train = data_setup::create_y(&cbb_full_data);
    let x_test = data_setup::create_x(&cbb_2020_data);
    println!("{:?}", x_train);
}