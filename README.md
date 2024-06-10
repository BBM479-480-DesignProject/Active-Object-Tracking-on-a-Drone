# Active-Object-Tracking-on-a-Drone
 
First you need visit airsim's website to install airsim with your system specifics.
https://microsoft.github.io/AirSim/

Then you need to recreate AI vehicles on the environment. Since Airsim uses Unreal Engine 4 you can follow a Unreal Engine 4 tutorial. 
https://www.youtube.com/watch?v=ICXrV9IXDVg&t=330s

For vehicle models you can use vehicle variety packs as we did. You can import the packages by just one click after downloading them.
https://www.unrealengine.com/marketplace/en-US/product/bbcb90a03f844edbb20c8b89ee16ea32
https://www.unrealengine.com/marketplace/en-US/product/9a705589d1994c6e8757fdbedaf698af

Here is the environments we used. You can install them by following the airsim's official documentation.
https://www.unrealengine.com/marketplace/en-US/product/vehicle-game
https://www.unrealengine.com/marketplace/en-US/product/landscape-mountains

After setting up everthing we can move on to airsim specifics.
First we need to change settings.json file on your computer. Go to documents/Airsim on your computer. Change settings.json with our's.

Then move our python files to Airsim/PythonClient/multirotor. 

After all these steps all you need to do is run the python file named HelloDrone.py and simulation.

Have fun.
