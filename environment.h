#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <memory>
#include <cmath>
#include <vector>
#include <QDebug>
#include "reinforcement_learning.h"
#include "utilities.h"

class Solarped {
public:
    Solarped(int action_dim, int state_dim)
        : action_dim(action_dim),
          state_dim(state_dim)
    {
        agent = std::make_shared<DQN>(action_dim, state_dim);
    };

    Solarped(int action_dim, int state_dim, State &init_state, State &goal_state)
        : action_dim(action_dim),
          state_dim(state_dim),
          current_state(init_state),
          goal_state(goal_state)
    {
        agent = std::make_shared<DQN>(action_dim, state_dim);
    };

    void initState(State &init_state) {
        setCurrentState(init_state);
    };

    void setGoalState(State &goal_state) {
        this->goal_state = goal_state;
    };

    State &getGoalState() {
        return goal_state;
    };

    void setCurrentState(State &current_state) {
        this->current_state = current_state;
    };

    State &getCurrentState() {
        return current_state;
    };

    double calculateReward(State &goal_state, State &next_state) {
        this->goal_state = goal_state;
        return calculateReward(next_state);
    };

    double calculateReward(State &next_state) {
        this->reward = -std::sqrt(std::pow(next_state.x-goal_state.x, 2)+std::pow(next_state.y-goal_state.y, 2)) - std::abs(next_state.angle-goal_state.angle);
        return this->reward;
    };

    int selectAction(State &current_state) {
        this->current_state = current_state;
        std::vector<double> current_state_vector;
        current_state_vector.push_back(current_state.x);
        current_state_vector.push_back(current_state.y);
        current_state_vector.push_back(current_state.angle);
        action = agent->selectAction(current_state_vector);
        return action;
    };

    int learn(State &goal_state, State &next_state) {
        this->goal_state = goal_state;
        return learn(next_state);
    };

    int learn(State &next_state) {
        qDebug() << "next state x: " << next_state.x;
        qDebug() << "next state y: " << next_state.y;
        qDebug() << "next state angle: " << next_state.angle;
        selectAction(next_state);
        calculateReward(next_state);
        storeExperience(next_state);
        current_state = next_state;
        if(agent->isMemoryFull())
            agent->learn();
        qDebug() << "action: " << action;
        return action;
    };

    void storeExperience(State &current_state, int action, double reward, State &next_state) {
        this->current_state = current_state;
        this->action = action;
        this->reward = reward;
        storeExperience(next_state);
    };

    void storeExperience(State &next_state) {
        std::vector<double> current_state_vector;
        current_state_vector.push_back(current_state.x);
        current_state_vector.push_back(current_state.y);
        current_state_vector.push_back(current_state.angle);

        std::vector<double> next_state_vector;
        next_state_vector.push_back(next_state.x);
        next_state_vector.push_back(next_state.y);
        next_state_vector.push_back(next_state.angle);

        agent->storeExperience(current_state_vector, action, reward, next_state_vector);
    };

    bool isDone(State &goal_state, State &next_state) {
        this->goal_state = goal_state;
        return isDone(next_state);
    };

    bool isDone(State &next_state) {
        double error = std::sqrt(std::pow(next_state.x-goal_state.x, 2)+std::pow(next_state.y-goal_state.y, 2)) + std::abs(next_state.angle-goal_state.angle);
        if(error < ERROR_TOLERANCE)
            return true;
        return false;
    };

    void saveAgent() {
        agent->saveModel();
    };

private:
    int action_dim;
    int state_dim;
    const double ERROR_TOLERANCE = 0.1;
    std::shared_ptr<DQN> agent;
    State goal_state;
    State current_state;
    int action;
    double reward;
};

#endif // ENVIRONMENT_H
