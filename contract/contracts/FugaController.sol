// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./PeerSystem.sol";

contract FugaController is PeerSystem {
    event ServerMessage(string field);

    enum Status{
        Ready,
        FitIns,
        FitRes,
        EvaluateIns,
        EvaluateRes
    }

    address owner;

    struct Config {
        uint batch_size;
        uint local_epochs;
        uint val_steps;
    }
    Config private config;
    
    uint private FitCount;
    uint private EvalCount;

    struct Model {
        Status status;
        string model_hash;
        uint num_samples;
        uint scores;
        uint updateCount;
        uint[] evals;
    }
    mapping(address => Model) private model;

    constructor() {
        owner=msg.sender;
    }

    modifier onlyOwner(){
        require(owner==msg.sender,"Only owner can call this function");
        _;
    }

    modifier checkStatus(Status _status){
        require(_status==model[msg.sender].status,"Status not match");
        _;
    }

    function FitIns() external onlyCurrentClients currentStage(Stage.Fit) checkStatus(Status.Ready) returns(string[] memory, uint[] memory, uint[] memory, uint, uint) {
        address[] memory peers = clientPeers[msg.sender];
        uint numPeers = peers.length;
        string[] memory model_hashes = new string[](numPeers);
        uint[] memory num_samples = new uint[](numPeers);
        uint[] memory scores = new uint[](numPeers);
        for(uint i=0; i<numPeers; i++) {
            model_hashes[i] = model[peers[i]].model_hash;
            num_samples[i] = model[peers[i]].num_samples;
            scores[i] = model[peers[i]].scores;
        }
        model[msg.sender].status = Status.FitIns;
        return (model_hashes, num_samples, scores, config.batch_size, config.local_epochs);
    }

    function FitRes(string memory _model_hash, uint _num_samples) external onlyCurrentClients currentStage(Stage.Fit) checkStatus(Status.FitIns) {
        model[msg.sender].model_hash = _model_hash;
        model[msg.sender].num_samples = _num_samples;
        model[msg.sender].updateCount++;
        FitCount++;
        model[msg.sender].status = Status.FitRes;
    }

    function EvaluateIns() external onlyCurrentClients currentStage(Stage.Evaluate) checkStatus(Status.FitRes) returns(string[] memory, uint, uint) {
        address[] memory peers = clientPeers[msg.sender];
        uint numPeers = peers.length;
        string[] memory model_hashes = new string[](numPeers);
        for(uint i=0; i<numPeers; i++) {
            model_hashes[i] = model[peers[i]].model_hash;
        }
        model[msg.sender].status = Status.EvaluateIns;
        return (model_hashes, config.batch_size, config.val_steps);
    }

    function EvaluateRes(string memory _model_hash, uint _num_samples) external onlyCurrentClients currentStage(Stage.Fit) checkStatus(Status.FitIns) {
        model[msg.sender].model_hash = _model_hash;
        model[msg.sender].num_samples = _num_samples;
        model[msg.sender].updateCount++;
        FitCount++;
        model[msg.sender].status = Status.FitRes;
    } 

    function finishRound() currentStage(Stage.EvalFinished) internal {
        delete currentClients;
        currentRound++;
        stage = Stage.Ready;
    } 

} 