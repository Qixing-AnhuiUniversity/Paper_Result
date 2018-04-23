# Paper_Result
  
  
   This is Qixing's github, which include the the experiment code and result of my research paper, with both my own method and the method used for comparision.
  
   
   If you find some errors in my statement, or if you are interest in my research, please contact with me. My email is 10031@ahu.edu.cn.
 
   
   In My first paper: "***Rotor resistance and excitation inductance estimation of induction motor using deep-q-learning algorithm***", I proposed an artificial intelligence method(Deep-Q Learning) to estimate induction motor's parameters. As Deep-Q Learning method is a data-based method, it has some advantages over some other traditional model-based strategies, such as Model Reference Adaption System(MRAS), Extenal Kalman Filter(EKF), Least Square Method(LSM) and so on.
  
   
   In order to justify it, four comparision experiments were set up : 1, IEEE standard 211, the most common method, which is very familiar to the engineer; 2, Multiobjective Particle Swarm Optimization(MOPSO), which is easy to implement and very effective; 3, A latest model-based method: Least Square Method with start-up Transient Measurement(LSMTM); 4, Deep-Q Learning method(DQL). Because IEEE standard 211 is quite common, I will not open source it. I just open source the other three method: MPSO, LSMSM and DQL. If the author of MPSO or LSMSM think it is inappropriate to open source it, or find some erros in the  source code, please contack with me, so I can delete it or revise it at your request.
   
   The experiment is implemented in my test bench, you can read anoter document "Experiment Setup.md" to find more details. However, as it is difficult to release a product-level source code in github, I only released some demo code based on my MATLAB/SIMULINK test bench. Also you can find some information of SIMULINK test bench in  "Experiment Setup.md".
   
   Here are  instructions about MOPSO:

    
1. Download **PSO.m** and **Sphere.m**
2. Open MATLAB, run **PSO.m**.
3. When running finised, open workspace, find **Gbest**, open it.
4. you can get the estimated rotor resistance value.

   

   
   
  And here are instructions about LSMTM:

1. Download **qxforLsmooth.mdl**, **lsmforpaper-1.py**, and **lsmdatacollect.m**
2. Run **qxforLsmooth.mdl** in Simulink. Noticed that my MATLAB version is MATLAB2016.b, Unbuntu. If you find some error when running, maybe there are two reasons: 1), your MATLAB verson must be 2016b or higher; 2) you have to run the model in Unbuntu system, as I haven't test it in Windows yet.
3. If step 2 finised successfully, run **lsmdatacollect.m**, the data will be stored in CSV format.
4. run **lsmforpaper-1.py**, finially you can get the right answer. (K1 and K2 calculated only, and you can transform it to Rr or Lm)
   
   
   

  (Revised on 4/23/2018) 
  
  Since my paper is accepted, I releasd the DQL algorithm. There are two files:**DQNmotor_action_32batchnewenv300.py** and **qxenv.mdl**. The running environment is: Unbuntu 16.04, matlab2016b and python2.7. All you need is to copy **qxenv.mdl** and **DQNmotor_action_32batchnewenv300.py** to the same directory and run **DQNmotor_action_32batchnewenv300.py** in python 2.7.
  
  I am busy these days for my new research work: ***Outlier detection of Induction motor***, which is very interesting but need hard work. When this work finishes, I will  submit two papers, one is focus on the application of outlier detection and the other is a pure theoretical paper which is about the generalization of Isolation Forest. After this, I wish I would have a rest, and I can give more details of Deep-Q-learning and Random Forest. 
