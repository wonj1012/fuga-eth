// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FugaController{
    event ServerMessage(string field);
    function MessageTest(string memory _field) external {
        emit ServerMessage(_field);
    }
    string[] model_hash = ['440dd98d6875144eace9694a56c57adc521efa464795f657f865b3288b6a7b09','8288d0fe3fb314d9430fdac3e1c5ccdc365cfca6ae24e2fb32acc8d14a2812d6','ca952603775ea03d204e95410aa14be3b5b638153ff1a413755e0b2693380ce6'];
    uint[] num_sample = [100, 100, 100];
    uint[] score = [100, 100, 100];

    function FitIns() external view returns(string[] memory, uint[] memory, uint[] memory, uint, uint) {
            uint numPeers = 3;
            string[] memory model_hashes = new string[](numPeers);
            uint[] memory num_samples = new uint[](numPeers);
            uint[] memory scores = new uint[](numPeers);
            for(uint i=0; i<numPeers; i++) {
                model_hashes[i] = model_hash[i];
                num_samples[i] = num_sample[i];
                scores[i] = score[i];
            }
            uint batch_size = 32;
            uint local_epochs = 1;
            return (model_hashes, num_samples, scores, batch_size, local_epochs);
        }
}