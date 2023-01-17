#include <random>
#include <chrono>
#include <QDebug>
#include <Eigen/Dense>
#include <iostream>
#include "environment.h"
#include "utilities.h"

int main()
{
    State goal_state;
    goal_state.x = 10;
    goal_state.y = 10;
    goal_state.angle = 100;

    State current_state;
    current_state.x = 0;
    current_state.y = 0;
    current_state.angle = 0;

    Solarped solarped(8, 3, current_state, goal_state);

    std::mt19937 seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::default_random_engine engine;
    std::uniform_real_distribution<float> generator;
    engine = std::default_random_engine(seed);
    generator = std::uniform_real_distribution<float>(0, 1);

    qDebug() << "start programe";

    for(int i = 0; i < 10000; i++) {
        current_state.x = 1080 * generator(engine);
        current_state.y = 720 * generator(engine);
        current_state.angle = 360 * generator(engine);
        auto start = std::chrono::high_resolution_clock::now();
        solarped.learn(current_state);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        qDebug() << "excution time: " << duration.count()*0.000001;
    }

    solarped.saveAgent();

    return 0;
}
