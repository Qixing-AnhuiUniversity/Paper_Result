# Experiment Setup #

Welcome to my blog:http://qixing-ahu.com/ , you can find more information on it.

In order to verify my theory, a motor test bench is neccesary. The test bench consist of following equipments:

1, two motors, which are called dynamometer motor and test motor respectively. The dynamometer motor is operated in speed mode, which means the motor keep its speed constant all the time, not affected by torque changes. The test motor is operated in torque, which means its electromagnetic is constant.
![](https://i.imgur.com/EA3v75q.jpg)
2，two motor controllers, one is used for dynamometer motor, and the other used for test motor.
![](https://i.imgur.com/gDlOFOe.jpg)
3, torque meter. used for measuring the test motor's torque.
![](https://i.imgur.com/ifU321Y.jpg)
4, data collector, used to record data from motor controllers and torque meter.
![](https://i.imgur.com/h52i8gU.jpg)
5, of course, you need a power source.

6, and the last, you need a computer for running the algorithm.

If you found it is hard to establish such an experiment environment, please don't worry. I will present a much simpler simulated environment. All you need is just a computer with the software of MATLAB2016b (or higher level version, for example, MATLAB2017a, etc), and, a python interpreter. 

 MATLAB2016b or higher version MATLAB is necessary because it has the API to python. you can combine MATLAB and python conveniently through MATLAB's API. More detail is availible from [https://cn.mathworks.com/help/matlab/apiref/matlab.engine.matlabengine-class.html](https://cn.mathworks.com/help/matlab/apiref/matlab.engine.matlabengine-class.html). 

MATLAB is used to simulate the motor test bench, and python is used to run the algorithm. From my DQN source code, you can see how to combine MATLAB with python, and how to work together.
