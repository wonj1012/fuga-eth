// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PeerSystem {

    enum Stage{
        Ready,
        Config,
        Fit,
        Evaluate,
        Finished
    }
    Stage internal stage;

    mapping(address => uint) internal clientRound;
    mapping(address => address[]) internal clientPeers;
    address[] internal currentClients;
    uint internal currentRound = 1;
    uint internal currentCompelete = 0;

    modifier onlyCurrentClients() {
        require(clientRound[msg.sender] == currentRound, "Client not in current round.");
        _;
    }
    
    modifier currentStage(Stage _stage){
        require(_stage==stage, "Current stage is not the expected stage.");
        _;
    }

    function joinRound() currentStage(Stage.Ready) external {
        require(clientRound[msg.sender] < currentRound, "Client already joined this round.");
        clientRound[msg.sender] = currentRound;
        currentClients.push(msg.sender);
    }

    function randPermutaion(address[] memory _arr) internal view returns (address[] memory) {
        uint n = _arr.length;
        address[] memory result = new address[](n);

        for (uint i = 0; i < n; i++) {
            result[i] = _arr[i];
        }

        for (uint i = n; i > 1; i--) {
            uint j = (uint(keccak256(abi.encodePacked(block.timestamp, i))) % i);
            (result[i - 1], result[j]) = (result[j], result[i - 1]);
        }

        return result;
    }

    function distributePeers(uint _numPeers) internal {
        require(currentClients.length > _numPeers, "Not enough clients to form peers.");
        address[] memory shuffledClients = randPermutaion(currentClients);
        uint numClients = currentClients.length;

        for (uint i = 0; i < numClients; i++) {
            address[] memory selectedPeers = new address[](_numPeers);

            for (uint j = 0; j < _numPeers; j++) {
                uint index = (i + j + 1) % numClients;
                selectedPeers[j] = shuffledClients[index];
            }

            clientPeers[currentClients[i]] = selectedPeers;
        }
    }

}

