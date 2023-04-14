// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FugaController{
    event ServerMessage(string field);
    function test() external {
        emit ServerMessage("test");
    }
}