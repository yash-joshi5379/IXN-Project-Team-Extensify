# Installing the UCL AnyConnect VPN on Linux devices
The VPN installation for Linux is not properly documented on the UCL website, so this guide will give clear steps on how to set up the VPN on Linux based devices.

1. Paste the following URL into a web browser:
```
https://www.ucl.ac.uk/isd/how-to/connecting-to-ucl-vpn-linux
```
 
2. Under the **AnyConnect client software instructions** section, click the **64bit Client software** link, and press the download icon (top left of window) to install the **OneDrive_1_04-03-2026 - linux x86_64 1.zip** folder.

3. Once the download is complete, open your local downloads folder and extract the zip folder. Open this new extracted folder, and you should see 2 **.tgz** files:
- cisco-secure-client-linux64-5.1.15.287-predeploy-deb-k9.tgz
- cisco-secure-client-linux64-5.1.15.287-predeploy-rpm-k9.tgz

4. For Ubuntu OS (Debian-based) devices, you only need the **predeploy-deb** folder, and you can ignore the **predeploy-rpm** folder completely. Double click the **cisco-secure-client-linux64-5.1.15.287-predeploy-deb-k9.tgz** file, and a folder with the same name should be created next to it. Check: open this newly created folder, and you should see these 7 files:
- cisco-secure-client-dart_5.1.15.287_amd64.deb
- cisco-secure-client-iseposture_5.1.15.287_amd64.deb
- cisco-secure-client-nvm_5.1.15.287_amd64.deb
- cisco-secure-client-posture_5.1.15.287_amd64.deb
- cisco-secure-client-vpn_5.1.15.287_amd64.deb
- cisco-secure-client-vpn-cli_5.1.15.287_amd64.deb
- CiscoSystemsInc.pgp

5. Now open a new terminal and run:
```
cd Downloads/OneDrive_1_04-03-2026\ -\ linux\ x86_64\ 1/cisco-secure-client-linux64-5.1.15.287-predeploy-deb-k9/
```

This will redirect you to inside the newly created folder. To check you are in the correct directory, run ```ls``` and you should see this:
```
cisco-secure-client-dart_5.1.15.287_amd64.deb
cisco-secure-client-iseposture_5.1.15.287_amd64.deb
cisco-secure-client-nvm_5.1.15.287_amd64.deb
cisco-secure-client-posture_5.1.15.287_amd64.deb
cisco-secure-client-vpn_5.1.15.287_amd64.deb
cisco-secure-client-vpn-cli_5.1.15.287_amd64.deb
CiscoSystemsInc.pgp
```

6. Now run this command:
```
sudo dpkg -i cisco-secure-client-vpn_5.1.15.287_amd64.deb
```

You should see this output:
```
Selecting previously unselected package cisco-secure-client-vpn.
(Reading database ... 193539 files and directories currently installed.)
Preparing to unpack cisco-secure-client-vpn_5.1.15.287_amd64.deb ...
Migrating /opt/cisco/anyconnect directory to /opt/cisco/secureclient directory
Unpacking cisco-secure-client-vpn (5.1.15.287) ...
Setting up cisco-secure-client-vpn (5.1.15.287) ...
Configuration file /usr/lib/systemd/system/vpnagentd.service is marked executable. Please remove executable permission bits. Proceeding anyway.
Created symlink /etc/systemd/system/multi-user.target.wants/vpnagentd.service → /usr/lib/systemd/system/vpnagentd.service.
Processing triggers for gnome-menus (3.36.0-1.1ubuntu3) ...
Processing triggers for desktop-file-utils (0.27-2build1) ...
Processing triggers for hicolor-icon-theme (0.17-2) ...
```

7. Now that the VPN client has been installed successfully, you should see the **Cisco Secure Client** in your app drawer. Open this client, and a connection window will appear. For the connection, enter this address: ```vpn.ucl.ac.uk```. Then click the **Connect** button.

8. A new window will appear, called **Cisco Secure Client - Login**. Now log in with your UCL credentials (same UCL email and password used to log in to Moodle). After a few seconds, the Cisco client will close and a pop-up will say **Connected**. Also, if you open the Cisco client again, it will say **Connected to vpn.ucl.ac.uk**.

9. To disconnect from the VPN, simply open the Cisco Secure Client and click the **Disconnect** button.
