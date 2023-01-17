#ifndef REINFORCEMENT_LEARNING_H
#define REINFORCEMENT_LEARNING_H

#include <torch/torch.h>
#include <vector>
#include <fstream>
#include <memory>
#include <random>
#include <chrono>
#include <QDebug>

#include "model.h"
#include "utilities.h"

class DQN {
public:
    DQN(int action_dim, int state_dim)
        : action_dim(action_dim),
          state_dim(state_dim),
          device(torch::kCUDA)
    {
        memory = std::make_shared<ExperienceMemory>(action_dim, state_dim);
        learn_step_counter = 0;
        if(torch::cuda::is_available()) {
            device_type = torch::kCUDA;
            qDebug() << "using GPU";
        } else {
            device_type = torch::kCPU;
        }
        device = torch::Device(device_type);

        model_evaluate = std::make_shared<Net>(state_dim, action_dim, HIDDEN_UNITES);
        model_target = std::make_shared<Net>(state_dim, action_dim, HIDDEN_UNITES);
        model_file.open("model.pt");
        if(model_file) {
            loadModel();
        }

        model_evaluate->to(device);
        model_target->to(device);
        model_optimizer = std::make_shared<torch::optim::Adam>(model_evaluate->parameters(), LEARNING_RATE);
        engine = std::default_random_engine(seed);
        generator_float = std::uniform_real_distribution<float>(0, 1);
        generator_action = std::uniform_real_distribution<float>(0, action_dim-1);
    };

    int selectAction(std::vector<double> &current_state_vector, bool add_noise=true) {
        current_state = torch::from_blob(current_state_vector.data(), {state_dim}).to(device);
        torch::Tensor actions_value;
        if(add_noise) {
            if(generator_float(engine) < EPSILON) {
                actions_value = model_evaluate->forward(current_state.unsqueeze(0)).to(device);
                action = torch::argmax(actions_value, 0).item().toInt();
            } else {
                action = (int)generator_action(engine);
            }
        } else {
            actions_value = model_evaluate->forward(current_state.unsqueeze(0)).to(device);
            action = torch::argmax(actions_value, 0).item().toInt();
        }

        return action;
    };

    double learn() {
        if(0 == learn_step_counter % TARGET_REPLACE_ITER)
            updateTargetNetwork();

        Experience experiences = memory->getExperiences(BATCH_SIZE);
        current_states = experiences.current_state;
        actions = experiences.action;
        rewards = experiences.reward;
        next_states = experiences.next_state;
        q_evaluate = model_evaluate->forward(current_states).to(device);
        q_next = model_target->forward(next_states).to(device);
        q_target = rewards + GAMMA * q_next;
        model_loss = torch::mse_loss(q_evaluate, q_target);
        model_optimizer->zero_grad();
        model_loss.backward();
        model_optimizer->step();

        learn_step_counter = learn_step_counter + 1;

        qDebug() << "loss: " << model_loss.item<double>();

        return model_loss.item<double>();
    };

    bool isMemoryFull() {
        return memory->isMemoryFull();
    };

    void storeExperience(std::vector<double> &current_state_vector, int action, double reward, std::vector<double> &next_state_vector) {
        torch::Tensor current_state_tensor = torch::from_blob(current_state_vector.data(), {state_dim}).to(device);
        torch::Tensor action_tensor = torch::mul(torch::ones(1), action).to(device);
        torch::Tensor reward_tensor = torch::mul(torch::ones(1), reward).to(device);
        torch::Tensor next_state_tensor = torch::from_blob(next_state_vector.data(), {state_dim}).to(device);

        memory->storeExperience(current_state_tensor, action_tensor, reward_tensor, next_state_tensor);
    };

    void saveModel() {
        torch::save(std::dynamic_pointer_cast<torch::nn::Module>(model_evaluate), "model.pt");
    };

    void loadModel() {
        torch::load(model_evaluate, "model.pt");
        torch::load(model_target, "model.pt");
    };

    ~DQN() {
        model_file.close();
    };

private:
    int action_dim;
    int state_dim;
    std::shared_ptr<ExperienceMemory> memory;
    int learn_step_counter;
    std::shared_ptr<Net> model_evaluate;
    std::shared_ptr<Net> model_target;
    std::ifstream model_file;
//    OUNoise noise;
    torch::DeviceType device_type;
//    std::shared_ptr<torch::Device> device;
    torch::Device device;
    std::shared_ptr<torch::optim::Adam> model_optimizer;

    torch::Tensor current_state;
    int action;

    torch::Tensor current_states;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor next_states;
    torch::Tensor q_evaluate;
    torch::Tensor q_next;
    torch::Tensor q_target;
    torch::Tensor model_loss;

    int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine;
    std::uniform_real_distribution<float> generator_float;
    std::uniform_real_distribution<float> generator_action;

    const int HIDDEN_UNITES = 50;
    const double LEARNING_RATE = 0.0001;
    const float EPSILON = 0.9;
    const int TARGET_REPLACE_ITER = 100;
    const int BATCH_SIZE = 128;
    const double GAMMA = 0.98;

    void updateTargetNetwork() {
        for(int i = 0; i < model_target->parameters().size(); i++) {
            model_target->parameters()[i] = model_evaluate->parameters()[i];
        }
    };
};

#endif // REINFORCEMENT_LEARNING_H
