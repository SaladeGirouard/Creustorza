import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import requests
from PIL import Image
import time
import hydralit_components as hc
import random
import numpy as np
from numpy import nan as Nan
import math

def request_api_video(tconst):
    url = f"https://api.themoviedb.org/3/movie/{tconst}?api_key=13ccd20d13e4468bc058a421a4c8987c&language=en-US"
  
    r = requests.get(url)
    config = r.json()    
    prefix = "https://www.youtube.com/watch?v="
    try:
        trailer = prefix + config['key']
    except:
        trailer = "/Users/utilisateur/Documents/streamlit/no_image.png"
    return trailer

### Importation de la base de données ###
films = pd.read_csv("https://raw.githubusercontent.com/robin0744/projet2/main/filmsv2_2.csv")


### Le visuel avec le logo et les rideaux ###
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""<img class='test' src="https://i.goopics.net/i7av9t.png" alt="Image">""",unsafe_allow_html=True)
with col2:
    st.image('logo_creustorza.png')
    titredufilm_annee = st.multiselect('', films["title_year"])
with col3:
    st.markdown("""<img src="https://i.goopics.net/lr3ws6.png" alt="Image">""",unsafe_allow_html=True)

    
    
def tconst_from_film(film_name):
    try: 
        tc = films.loc[films["title_year"] == film_name]["tconst"].values[0]
    except:
        tc = "coucou"
    return tc

 
def request_api(tconst):
    url = f"https://api.themoviedb.org/3/movie/{tconst}?api_key=13ccd20d13e4468bc058a421a4c8987c&language=en-US"

    r = requests.get(url)
    config = r.json()    
    prefix = "https://image.tmdb.org/t/p/original"
    try:
        img = prefix + config['backdrop_path']
    except:
        img = "pas_dimage.png"
    return img


##################################################
### LA FONCTION AVEC LES FILMS PROCHES NORMAUX ###
##################################################

def filmsprochesbasique(titredufilm_annee):
    
    # On met le titre du film en minuscules :
 
    titredufilm_annee = ' '.join(titredufilm_annee)
        # On créé la variable X qui contient les critères sur lesquels nous faisons
        # notre recherche.
    X = films[['startYear','averageRating',
              'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
              'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western',"numVoteslog"]]


        # On standardise nos variables dans X : on leurs donne toutes le même poids. 
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

        # On entraîne notre modèle, sur une base de "NearestNeighbors", pour avoir 15
        # résultats, et on met un poids plus important sur la "distance"
    distanceKNN = NearestNeighbors(n_neighbors=15).fit(X_scaled)

        # On entre le nom du film recherché (titredufilm) dans notre modèle entraîné.
        # Le résultat, appelé "tuplevoisin", est un tuple de deux éléments : deux 
        # array numpys, dont le second contient les index des films-résultats de la
        # requête
    #films['title_year'] = films['title_year'].apply(lambda x: x.lower())
    tuplevoisin = distanceKNN.kneighbors(scaler.transform(films.loc[films['title_year'] == titredufilm_annee,
                                                                        ['startYear','averageRating',
                                                                        'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                                                                        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
                                                                        'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                                                                        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'numVoteslog']]))
    
    # On créé un DataFrame vide, dans lequel on va mettre les films-résultats
    dffilmproches = pd.DataFrame()

    # Par une boucle "for", dont le cycle dure autant que le nombre de résultats (ici 15),
    # on ajoute, un par un, les films-résultats ainsi que les infos spécifiées
    for index in range(0,len(tuplevoisin[1][0])):
        dffilmproches=dffilmproches.append(films.loc[films.index == (tuplevoisin[1][0][index])].iloc[0], ignore_index=True)

    dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)

    #Il faut mettre startYear et numVotes en int sinon ce sont des floats
    dffilmproches["startYear"]=dffilmproches["startYear"].apply(lambda x: int(x))
    dffilmproches["numVotes"]=dffilmproches["numVotes"].apply(lambda x: int(x))
    
    # la fonction retourne le DataFrame composé des films-résultats
    return dffilmproches[["title_year","genres", "averageRating", "numVotes"]]


####################################
### LA FONCTION AVEC LES NANARDS ###
####################################


def filmsprochesnanards(titredufilm_annee):
  df_nanards = films.loc[films['averageRating']<3.5]
  titredufilm_annee = ' '.join(titredufilm_annee)
  #films['title_year'] = films['title_year'].apply(lambda x: x.lower())

  df_nanards2 = pd.concat([df_nanards,films.loc[films['title_year'] == titredufilm_annee]], ignore_index=True)
  df_nanards2.drop_duplicates(subset=["title_year"], inplace=True)

  df_nanards2.reset_index(inplace=True)

  X = df_nanards2[['startYear',
              'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
              'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']]
  scaler = StandardScaler().fit(X)
  X_scaled = scaler.transform(X)
  distanceKNN = NearestNeighbors(n_neighbors=15).fit(X_scaled)
  tuplevoisin = distanceKNN.kneighbors(scaler.transform(df_nanards2.loc[df_nanards2['title_year'] == titredufilm_annee,
                                                                        ['startYear',
                                                                        'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                                                                        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
                                                                        'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                                                                        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']]))
  # On créé un DataFrame vide, dans lequel on va mettre les films-résultats
  dffilmproches = pd.DataFrame()
    # Par une boucle "for", dont le cycle dure autant que le nombre de résultats (ici 15),
    # on ajoute, un par un, les films-résultats ainsi que les infos spécifiées
  for index in range(0,len(tuplevoisin[1][0])):
    dffilmproches=dffilmproches.append(df_nanards2.loc[df_nanards2.index == (tuplevoisin[1][0][index])].iloc[0], ignore_index=True)
    #Il faut mettre startYear et numVotes en int sinon ce sont des floats
  
  dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)


  dffilmproches["startYear"]=dffilmproches["startYear"].apply(lambda x: int(x))
  dffilmproches["numVotes"]=dffilmproches["numVotes"].apply(lambda x: int(x))
    # la fonction retourne le DataFrame composé des films-résultats
  return dffilmproches[["title_year","genres", "averageRating", "numVotes"]]


###############################################
### LA FONCTION AVEC LES BONS FILMS PROCHES ###
###############################################


def bonsfilmsproches(titredufilm_annee):

  df_bestfilm = films.loc[(films['averageRating']>7.5)&(films['numVotes']>50)]
  titredufilm_annee = ' '.join(titredufilm_annee)
  #films['title_year'] = films['title_year'].apply(lambda x: x.lower())
  df_bestfilm2 = pd.concat([df_bestfilm,films.loc[films['title_year'] == titredufilm_annee]], ignore_index=True)
  df_bestfilm2.drop_duplicates(subset=["title_year"], inplace=True)

  df_bestfilm2.reset_index(inplace=True)

  X = df_bestfilm2[['startYear', 'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
              'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'numVoteslog']]
  scaler = StandardScaler().fit(X)
  X_scaled = scaler.transform(X)
  distanceKNN = NearestNeighbors(n_neighbors=15).fit(X_scaled)
  tuplevoisin = distanceKNN.kneighbors(scaler.transform(df_bestfilm2.loc[df_bestfilm2['title_year'] == titredufilm_annee,
                                                                        ['startYear', 'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
              'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'numVoteslog']]))
  # On créé un DataFrame vide, dans lequel on va mettre les films-résultats
  dffilmproches = pd.DataFrame()
    # Par une boucle "for", dont le cycle dure autant que le nombre de résultats (ici 15),
    # on ajoute, un par un, les films-résultats ainsi que les infos spécifiées
  for index in range(0,len(tuplevoisin[1][0])):
    dffilmproches=dffilmproches.append(df_bestfilm2.loc[df_bestfilm2.index == (tuplevoisin[1][0][index])].iloc[0], ignore_index=True)

  dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)
  
    #Il faut mettre startYear et numVotes en int sinon ce sont des floats
  dffilmproches["startYear"]=dffilmproches["startYear"].apply(lambda x: int(x))
  dffilmproches["numVotes"]=dffilmproches["numVotes"].apply(lambda x: int(x))
  return dffilmproches[["title_year","genres", "averageRating", "numVotes"]]


#####################################################
### LA FONCTION AVEC LES FILMS PROCHES PAS CONNUS ###
#####################################################

def filmsprochespasconnus(titredufilm_annee):
    
    # On met le titre du film en minuscules :
    filmspasconnus = films.loc[films["numVotes"]<4000]
    titredufilm_annee = ' '.join(titredufilm_annee)
    #films['title_year'] = films['title_year'].apply(lambda x: x.lower())

    filmspasconnus2 = pd.concat([filmspasconnus,films.loc[films['title_year'] == titredufilm_annee]], ignore_index=True)
    filmspasconnus2.drop_duplicates(subset=["title_year"], inplace=True)

    filmspasconnus2.reset_index(inplace=True)

        # On créé la variable X qui contient les critères sur lesquels nous faisons
        # notre recherche.
    X = filmspasconnus2[['startYear','averageRating',
              'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
              'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']]


    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

        # On entraîne notre modèle, sur une base de "NearestNeighbors", pour avoir 15
        # résultats, et on met un poids plus important sur la "distance"
    distanceKNN = NearestNeighbors(n_neighbors=15).fit(X_scaled)

        # On entre le nom du film recherché (titredufilm) dans notre modèle entraîné.
        # Le résultat, appelé "tuplevoisin", est un tuple de deux éléments : deux 
        # array numpys, dont le second contient les index des films-résultats de la
        # requête
    tuplevoisin = distanceKNN.kneighbors(scaler.transform(filmspasconnus2.loc[filmspasconnus2['title_year'] == titredufilm_annee,
                                                                        ['startYear','averageRating',
                                                                        'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                                                                        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
                                                                        'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                                                                        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']]))
    
    # On créé un DataFrame vide, dans lequel on va mettre les films-résultats
    dffilmproches = pd.DataFrame()

    # Par une boucle "for", dont le cycle dure autant que le nombre de résultats (ici 15),
    # on ajoute, un par un, les films-résultats ainsi que les infos spécifiées
    for index in range(0,len(tuplevoisin[0][0])):
        dffilmproches=dffilmproches.append(filmspasconnus2.loc[filmspasconnus2.index == (tuplevoisin[1][0][index])].iloc[0], ignore_index=True)

    dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)

    #Il faut mettre startYear et numVotes en int sinon ce sont des floats
    dffilmproches["startYear"]=dffilmproches["startYear"].apply(lambda x: int(x))
    dffilmproches["numVotes"]=dffilmproches["numVotes"].apply(lambda x: int(x))
    
    # la fonction retourne le DataFrame composé des films-résultats
    return dffilmproches[["title_year","genres", "averageRating", "numVotes"]]


##################################################
### LA FONCTION AVEC LES FILMS PROCHES RECENTS ###
##################################################


def filmsprochesrecents(titredufilm_annee):
    
    # On met le titre du film en minuscules :
    filmsrecents = films.loc[films["startYear"]>=2015]
    titredufilm_annee = ' '.join(titredufilm_annee)
    #films['title_year'] = films['title_year'].apply(lambda x: x.lower())

    filmsrecents2 = pd.concat([filmsrecents,films.loc[films['title_year'] == titredufilm_annee]], ignore_index=True)
    filmsrecents2.drop_duplicates(subset=["title_year"], inplace=True)

    filmsrecents2.reset_index(inplace=True)

        # On créé la variable X qui contient les critères sur lesquels nous faisons
        # notre recherche.
    X = filmsrecents2[['averageRating',
              'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
              'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'numVoteslog']]


    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

        # On entraîne notre modèle, sur une base de "NearestNeighbors", pour avoir 15
        # résultats, et on met un poids plus important sur la "distance"
    distanceKNN = NearestNeighbors(n_neighbors=15).fit(X_scaled)

        # On entre le nom du film recherché (titredufilm) dans notre modèle entraîné.
        # Le résultat, appelé "tuplevoisin", est un tuple de deux éléments : deux 
        # array numpys, dont le second contient les index des films-résultats de la
        # requête
    tuplevoisin = distanceKNN.kneighbors(scaler.transform(filmsrecents2.loc[filmsrecents2['title_year'] == titredufilm_annee,
                                                                        ['averageRating',
                                                                        'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                                                                        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
                                                                        'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                                                                        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'numVoteslog']]))
    
    # On créé un DataFrame vide, dans lequel on va mettre les films-résultats
    dffilmproches = pd.DataFrame()

    # Par une boucle "for", dont le cycle dure autant que le nombre de résultats (ici 15),
    # on ajoute, un par un, les films-résultats ainsi que les infos spécifiées
    for index in range(0,len(tuplevoisin[0][0])):
        dffilmproches=dffilmproches.append(filmsrecents2.loc[filmsrecents2.index == (tuplevoisin[1][0][index])].iloc[0], ignore_index=True)

    dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)

    #Il faut mettre startYear et numVotes en int sinon ce sont des floats
    dffilmproches["startYear"]=dffilmproches["startYear"].apply(lambda x: int(x))
    dffilmproches["numVotes"]=dffilmproches["numVotes"].apply(lambda x: int(x))
    
    # la fonction retourne le DataFrame composé des films-résultats
    return dffilmproches[["title_year","genres", "averageRating", "numVotes"]]


##################################################
### LA FONCTION AVEC LES FILMS PROCHES ANCIENS ###
##################################################


def filmsprochesanciens(titredufilm_annee):
    # On met le titre du film en minuscules :
    filmsanciens = films.loc[films["startYear"]<1980]
    titredufilm_annee = ' '.join(titredufilm_annee)
    #films['title_year'] = films['title_year'].apply(lambda x: x.lower())

    filmsanciens2 = pd.concat([filmsanciens,films.loc[films['title_year'] == titredufilm_annee]], ignore_index=True)
    filmsanciens2.drop_duplicates(subset=["title_year"], inplace=True)

    filmsanciens2.reset_index(inplace=True)

        # On créé la variable X qui contient les critères sur lesquels nous faisons
        # notre recherche.
    X = filmsanciens2[['averageRating',
              'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
              'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
              'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'numVoteslog']]


    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

        # On entraîne notre modèle, sur une base de "NearestNeighbors", pour avoir 15
        # résultats, et on met un poids plus important sur la "distance"
    distanceKNN = NearestNeighbors(n_neighbors=15).fit(X_scaled)

        # On entre le nom du film recherché (titredufilm) dans notre modèle entraîné.
        # Le résultat, appelé "tuplevoisin", est un tuple de deux éléments : deux 
        # array numpys, dont le second contient les index des films-résultats de la
        # requête
    tuplevoisin = distanceKNN.kneighbors(scaler.transform(filmsanciens2.loc[filmsanciens2['title_year'] == titredufilm_annee,
                                                                        ['averageRating',
                                                                        'Action', 'Adult', 'Adventure', 'Animation', 'Biography', 'Comedy',
                                                                        'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'Film-Noir',
                                                                        'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
                                                                        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western', 'numVoteslog']]))
    
    # On créé un DataFrame vide, dans lequel on va mettre les films-résultats
    dffilmproches = pd.DataFrame()

    # Par une boucle "for", dont le cycle dure autant que le nombre de résultats (ici 15),
    # on ajoute, un par un, les films-résultats ainsi que les infos spécifiées
    for index in range(0,len(tuplevoisin[0][0])):
        dffilmproches=dffilmproches.append(filmsanciens2.loc[filmsanciens2.index == (tuplevoisin[1][0][index])].iloc[0], ignore_index=True)

    dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)

    #Il faut mettre startYear et numVotes en int sinon ce sont des floats
    dffilmproches["startYear"]=dffilmproches["startYear"].apply(lambda x: int(x))
    dffilmproches["numVotes"]=dffilmproches["numVotes"].apply(lambda x: int(x))
    
    # la fonction retourne le DataFrame composé des films-résultats
    return dffilmproches[["title_year","genres", "averageRating", "numVotes"]]



def listealeatoire():
  liste = []
  while len(liste) < 3 :
    a = random.randint(0,13)
    if a in liste:
      continue
    else :
      liste.append(a)

  return liste

def listealeatoirePeople(df):
  liste = []

  while len(liste) < 3 :
    a = random.randint(0,len(df.index)-3)
    if a in liste:
      continue
    else :
      liste.append(a)

#   return liste

#######################################################
### LA FONCTION AVEC LES FILMS DE LA REALISATEURICE ###
#######################################################


def filmsreal(titredufilm_annee):
    films['directorsName'] = films['directorsName'].apply(lambda x: eval(x))
    titredufilm_annee = ' '.join(titredufilm_annee)
    realisateur = films["directorsName"].loc[films["title_year"]==titredufilm_annee].iloc[0][0]
    dffilmproches = films.loc[films['directorsNamestr'].str.contains(realisateur)]

    dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)

#     if len(dffilmproches.index) > 3:
#         liste3=listealeatoirePeople(dffilmproches)
#         dfacteursaleat = pd.DataFrame()
#         for index in range(0,3):
#             dfacteursaleat = dfacteursaleat.append(dffilmproches.iloc[liste3[index]])
#         dffilmproches = dfacteursaleat



    vide = pd.Series([Nan,Nan,"Pas d'autre film",Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan], 
                    index=['Unnamed: 0', 'tconst', 'title_year', 'title_min', 'genres',
                'startYear', 'averageRating', 'numVotes', 'numVoteslog',
                'runtimeMinutes', 'Action', 'Adult', 'Adventure', 'Animation',
                'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
                'Mystery', 'News', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller',
                'War', 'Western', 'actorsName', 'actorsNamestr', 'directorsName',
                'directorsNamestr'])

    dffilmproches=dffilmproches.append(vide, ignore_index=True)
    dffilmproches=dffilmproches.append(vide, ignore_index=True)
    dffilmproches=dffilmproches.append(vide, ignore_index=True)

    return dffilmproches[["title_year","genres", "averageRating", "numVotes", "directorsName","actorsName"]]


#################################################
### LA FONCTION AVEC LES FILMS DE L'ACTEURICE ###
#################################################


def filmsacteur(titredufilm_annee):
    titredufilm_annee = ' '.join(titredufilm_annee)
    films['actorsName'] = films['actorsName'].apply(lambda x: eval(x))
    acteur = films["actorsName"].loc[films["title_year"]==titredufilm_annee].iloc[0][0]
    dffilmproches = films.loc[films['actorsNamestr'].str.contains(acteur)]

    dffilmproches.drop(dffilmproches.loc[dffilmproches['title_year'] == titredufilm_annee].index, inplace =True)


#     if len(dffilmproches.index) > 3:
#         liste3=listealeatoirePeople(dffilmproches)
#         dfacteursaleat = pd.DataFrame()
#         for index in range(0,3):
#             dfacteursaleat = dfacteursaleat.append(dffilmproches.iloc[liste3[index]])
#         dffilmproches = dfacteursaleat


    vide = pd.Series([Nan,Nan,"Pas d'autre film",Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan], 
                    index=['Unnamed: 0', 'tconst', 'title_year', 'title_min', 'genres',
                'startYear', 'averageRating', 'numVotes', 'numVoteslog',
                'runtimeMinutes', 'Action', 'Adult', 'Adventure', 'Animation',
                'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
                'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical',
                'Mystery', 'News', 'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller',
                'War', 'Western', 'actorsName', 'actorsNamestr', 'directorsName',
                'directorsNamestr'])

    dffilmproches=dffilmproches.append(vide, ignore_index=True)
    dffilmproches=dffilmproches.append(vide, ignore_index=True)
    dffilmproches=dffilmproches.append(vide, ignore_index=True)


    return dffilmproches[["title_year","genres", "averageRating", "numVotes", "directorsName","actorsName"]]


#######################################################
### LA FONCTION PRINCIPALE QUI CONCATENE LES AUTRES ###
#######################################################


def filmsproches(titredufilm_annee):
    df_creustorza = pd.DataFrame()

    dfbasique = filmsprochesbasique(titredufilm_annee)
#     liste1=listealeatoire()
#     dfbasiquealeatoire = pd.DataFrame()
#     dfbasiquealeatoire = dfbasiquealeatoire.append(dfbasique.iloc[liste1[0]])
#     dfbasiquealeatoire = dfbasiquealeatoire.append(dfbasique.iloc[liste1[1]])
#     dfbasiquealeatoire = dfbasiquealeatoire.append(dfbasique.iloc[liste1[2]])
    df_creustorza = pd.concat([df_creustorza,dfbasique], ignore_index=True)

    dfnanards = filmsprochesnanards(titredufilm_annee).head(3)
    df_creustorza = pd.concat([df_creustorza,dfnanards], ignore_index=True)

    dfbonsfilms = bonsfilmsproches(titredufilm_annee)
#     liste2=listealeatoire()
#     dfbonsfilmsaleatoire = pd.DataFrame()
#     dfbonsfilmsaleatoire = dfbonsfilmsaleatoire.append(dfbonsfilms.iloc[liste2[0]])
#     dfbonsfilmsaleatoire = dfbonsfilmsaleatoire.append(dfbonsfilms.iloc[liste2[1]])
#     dfbonsfilmsaleatoire = dfbonsfilmsaleatoire.append(dfbonsfilms.iloc[liste2[2]])
    df_creustorza = pd.concat([df_creustorza,dfbonsfilms], ignore_index=True)

    dfpasconnus = filmsprochespasconnus(titredufilm_annee).head(3)
    df_creustorza = pd.concat([df_creustorza,dfpasconnus], ignore_index=True)

    dfrecents = filmsprochesrecents(titredufilm_annee).head(3)
    df_creustorza = pd.concat([df_creustorza,dfrecents], ignore_index=True)

    dfanciens = filmsprochesanciens(titredufilm_annee).head(3)
    df_creustorza = pd.concat([df_creustorza,dfanciens], ignore_index=True)

    dfacteur = filmsacteur(titredufilm_annee).head(3)
    df_creustorza = pd.concat([df_creustorza,dfacteur], ignore_index=True)

    dfreal = filmsreal(titredufilm_annee).head(3)
    df_creustorza = pd.concat([df_creustorza,dfreal], ignore_index=True)

    return df_creustorza


### FONCTION POUR AVOIR LE NOM DE L'ACTEURICE PRINCIPALE EN FONCTION DU TITRE DU FILM
def nomacteur(titredufilm_annee):
    acteur = films["actorsName"].loc[films["title_year"]==titredufilm_annee[0]].iloc[0][0]
    return acteur

### FONCTION POUR AVOIR LE NOM DU/DE LA REALISATEURICE EN FONCTION DU TITRE DU FILM
def nomreal(titredufilm_annee):
  return films["directorsName"].loc[films["title_year"]==titredufilm_annee[0]].iloc[0][0]


############################
### LE RENDU DU RESULTAT ###
############################

if titredufilm_annee:
    with hc.HyLoader('2 secondes, ça mouline...',hc.Loaders.standard_loaders,index=[5]):
        time.sleep(10)
    st.markdown(
        """
        <style>
            .test{
                padding-right: 82px;
            }
            .big-font { font-size:32px !important; }
        </style>
        """,
        unsafe_allow_html=True)

    st.markdown("<p class='big-font'>Le Creus'torza vous propose</p>", unsafe_allow_html=True)
    if len(titredufilm_annee) < 2:
        result_film = filmsproches(titredufilm_annee)
    else:
        st.warning("You have to select only 1 movie")
    result_film["tconst"] = result_film["title_year"].apply(tconst_from_film)
    name_film = result_film["title_year"].to_list()
    url_img = result_film["tconst"].apply(request_api).to_list()
    url_video = result_film["tconst"].apply(request_api_video).to_list()
    acteur = nomacteur(titredufilm_annee)
    real =  nomreal(titredufilm_annee)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[0], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[0]}**')

    with col2:
        st.image(url_img[1], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[1]}**')

    with col3:
        st.image(url_img[2], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[2]}**')
    
    st.markdown("<p class='big-font'>Cultivez vos propres navets</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[3], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[3]}**')

    with col2:
        st.image(url_img[4], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[4]}**')

    with col3:
        st.image(url_img[5], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[5]}**')
        
    st.markdown("<p class='big-font'>Les films les mieux notés </p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[6], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[6]}**')

    with col2:
        st.image(url_img[7], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[7]}**')

    with col3:
        st.image(url_img[8], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[8]}**')

    st.markdown("<p class='big-font'>Vous ne les connaissez peut-être pas</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[9], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[9]}**')

    with col2:
        st.image(url_img[10], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[10]}**')

    with col3:
        st.image(url_img[11], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[11]}**')
        
    st.markdown("<p class='big-font'>Les films les plus récents mais pas trop</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[12], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[12]}**')

    with col2:
        st.image(url_img[13], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[13]}**')

    with col3:
        st.image(url_img[14], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[14]}**')
        
    st.markdown("<p class='big-font'>Un brin nostalgique ?</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[15], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[15]}**')

    with col2:
        st.image(url_img[16], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[16]}**')

    with col3:
        st.image(url_img[17], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[17]}**')
    st.markdown(f"<p class='big-font'>D'autres films de {acteur}</p>" , unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[18], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[18]}**')

    with col2:
        st.image(url_img[19], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[19]}**')

    with col3:
        st.image(url_img[20], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[20]}**')
        
    st.markdown(f"<p class='big-font'>D'autres films de {real}</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(url_img[21], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[21]}**')

    with col2:
        st.image(url_img[22], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[22]}**')

    with col3:
        st.image(url_img[23], width= 270, use_column_width= 'always')
        st.markdown(f'**{name_film[23]}**')
