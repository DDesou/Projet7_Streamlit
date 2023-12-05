# OC_Project7
## Projet OpenClassroom parcours Data Scientist

### Description du dossier Streamlit
L'interface utilisateur est une application Streamlit.  
En plus du dossier (fichier Yaml) de workflows, décrivant les différentes tâches à effectuer, 3 fichiers seulement sont nécessaires :
- **le script de l'application nommé 'dashboard.py'** détaillé plus bas
- **les fichiers png du logo de la société fictive et celui d'OCR**
- **le fichier des requirements** limités à l'application
- **/!\ A noter qu'un script de lancement de l'API sous Azure est nécessaire (Configuration/General Settings/Startup command)** : 
> python -m streamlit run dashboard.py --server.port 8000 --server.address 0.0.0.0

A noter que le dashboard doit être installé sous une **Web App Azure Linux en Python 3.11**.
L'URL de l'application est la suivante : **https://basicwebappvl.azurewebsites.net**. Bien entendu, pour qu'elle fonctionne, elle nécessite d'être activée AINSI QUE l'API!

Description de l'expérience utilisateur pour un client :
- le conseiller est invité à sélectionner le numéro d'un client. Un score de prédiction d'appartenance à la classe 0 (solvable) est ainsi affiché (sous forme de gauge plot, par rapport au seuil déterminé de 60%). POur ledit client, il est possible de sétectionner une feature choisie et de voir qu'elle est la valeur de cette valeur sur un graphique de 'density plots' des 2 groupes de population (classes 0 et 1). De même, les résultats des shap values (interprétabilité locale) nous renseigne sur les 10 variables qui 'tirent' le plus l'individu en question vers le groupe des 0 ou celui des 1 (non solvables). 

### Wworkflows
La plateforme d'hébergement choisie est **Azure Web App**.  
Afin de mettre en place un **processus d'intégration/amélioration continues**, le code est hébergé sur des **repo Git distants** et le déploiement réalisé par les **actions Github** communiquant avec l'hébergeur. De cette manière, des modifications peuvent être réalisées puis contrôlées d'abord dans un **environnement virtuel local** défini, puis éventuellement déployées dans une **nouvelle branche** avant d'être envoyées à la branche principale.  
Il a été décidé de séparer complètement le déploiement de l'API de celui de l'application et des projets Github distincts ont été créés :
- pour l'API : https://github.com/DDesou/Projet7_VL
- pour l'interface utilisateur : https://github.com/DDesou/Projet7_Streamlit