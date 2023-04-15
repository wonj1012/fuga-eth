// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./PeerSystem.sol";

contract FugaController is PeerSystem {
    event ServerMessage(string field);

    enum Status{
        Ready,
        Config,
        FitIns,
        FitRes,
        EvaluateIns,
        EvaluateRes
    }

    struct Config {
        bool self_centered;
        uint batch_size;
        uint learning_rate;
        uint local_epochs;
        uint val_steps;
    }
    Config private config;

    struct Client {
        Status status;
        string model_hash;
        uint num_samples;
        uint scores;
        uint updateCount;
        uint[] evals;
    }
    mapping(address => Client) private client;

    modifier checkStatus(Status _status){
        require(_status==client[msg.sender].status,"Status not match");
        _;
    }

    function getConfig() onlyCurrentClients currentStage(Stage.Ready) checkStatus(Status.Ready) external view returns(bool, uint, uint, uint, uint){
        return (config.self_centered, config.batch_size, config.learning_rate, config.local_epochs, config.val_steps);
    }

    function setConfig(bool _self_centered, uint _batch_size, uint _learning_rate, uint _local_epochs, uint _val_steps) private {
        config.self_centered = _self_centered;
        config.batch_size = _batch_size;
        config.learning_rate = _learning_rate;
        config.local_epochs = _local_epochs;
        config.val_steps = _val_steps;
    }

    function ConfigRes() external onlyCurrentClients currentStage(Stage.Ready) checkStatus(Status.Ready) {
        client[msg.sender].status = Status.Config;
        currentCompelete++;
        if(currentCompelete == currentClients.length){
            stage = Stage.Fit;
            currentCompelete = 0;
            emit ServerMessage("FitIns");
        }
    }

    function FitIns() external onlyCurrentClients currentStage(Stage.Fit) checkStatus(Status.Config) returns(string[] memory, uint[] memory, uint[] memory) {
        address[] memory peers = clientPeers[msg.sender];
        uint numPeers = peers.length;
        string[] memory model_hashes = new string[](numPeers);
        uint[] memory num_samples = new uint[](numPeers);
        uint[] memory scores = new uint[](numPeers);
        for(uint i=0; i<numPeers; i++) {
            model_hashes[i] = client[peers[i]].model_hash;
            num_samples[i] = client[peers[i]].num_samples;
            scores[i] = client[peers[i]].scores;
        }
        client[msg.sender].status = Status.FitIns;
        return (model_hashes, num_samples, scores);
    }

    function FitRes(string memory _model_hash, uint _num_samples) external onlyCurrentClients currentStage(Stage.Fit) checkStatus(Status.FitIns) {
        client[msg.sender].model_hash = _model_hash;
        client[msg.sender].num_samples = _num_samples;
        client[msg.sender].updateCount++;
        client[msg.sender].status = Status.FitRes;
        currentCompelete++;
        if(currentCompelete == currentClients.length){
            stage = Stage.Evaluate;
            currentCompelete = 0;
            emit ServerMessage("EvaluateIns");
        }
    }

    function EvaluateIns() external onlyCurrentClients currentStage(Stage.Evaluate) checkStatus(Status.FitRes) returns(string[] memory) {
        uint numPeers = clientPeers[msg.sender].length;
        string[] memory model_hashes = new string[](numPeers);
        for(uint i=0; i<numPeers; i++) {
            model_hashes[i] = client[clientPeers[msg.sender][i]].model_hash;
        }
        delete client[msg.sender].evals;
        client[msg.sender].status = Status.EvaluateIns;
        return (model_hashes);
    }

    function EvaluateRes(string[] memory _model_hashes, uint[] memory _values) external onlyCurrentClients currentStage(Stage.Evaluate) checkStatus(Status.EvaluateIns) {
        uint numPeers = _model_hashes.length;
        for(uint i=0; i<numPeers; i++) {
            for(uint j=0; j<numPeers; j++) {
                Client storage peer = client[clientPeers[msg.sender][(i+j)%numPeers]];
                if(keccak256(abi.encodePacked(peer.model_hash)) == keccak256(abi.encodePacked(_model_hashes[i]))) {
                    peer.evals.push(_values[i]);
                    continue;
                }
            }
        }
        client[msg.sender].status = Status.EvaluateRes;
        if(currentCompelete == currentClients.length){
            stage = Stage.Finished;
            currentCompelete = 0;
            finishRound();
        }
    } 


    function sort(uint[] memory data) internal pure returns (uint[] memory) {
        // Bubble sort the data in ascending order
        uint n = data.length;
        for (uint i = 0; i < n; i++) {
            for (uint j = i+1; j < n; j++) {
                if (data[i] > data[j]) {
                    uint temp = data[i];
                    data[i] = data[j];
                    data[j] = temp;
                }
            }
        }
        return data;
    }

    function getMedian(uint[] memory data) public pure returns (uint) {
        // Calculate the median of the data
        uint n = data.length;
        uint[] memory sortedData = sort(data);
        uint median = n % 2 == 0 ? (sortedData[n/2 - 1] + sortedData[n/2]) / 2 : sortedData[n/2];

        return median;
    }

    function normaScore() private {
        uint totalScore = 0;
        for(uint i=0; i<currentClients.length; i++) {
            totalScore += client[currentClients[i]].scores;
        }
        for(uint i=0; i<currentClients.length; i++) {
            client[currentClients[i]].scores = client[currentClients[i]].scores * 100 / totalScore;
        }
    }

    function finishRound() currentStage(Stage.Finished) private {
        for(uint i=0; i<currentClients.length; i++) {
            uint[] memory evals = client[currentClients[i]].evals;
            uint median = getMedian(evals);
            client[currentClients[i]].scores = median;
        }
        delete currentClients;
        currentRound++;
        stage = Stage.Ready;
    } 

} 