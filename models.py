# SQLAlchemy-related Imports
import sqlalchemy as db
from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


class ImageSQL:
    """
        Be able to:
        1) Return all images from Database
        2) Check if ImageNameExists
        3) Insert Image
        4) Update URL
        5) Update Caption
        6) Delete Image
    """
    Base = declarative_base()

    class Image(Base):
        __tablename__ = 'Image'
        ID = db.Column(db.INTEGER, primary_key=True)
        name = db.Column(db.VARCHAR(255))
        image_url = db.Column(db.VARCHAR(255))
        caption = db.Column(db.VARCHAR(255))

    def __init__(self, username, password, ip, port, database_name):
        inner_engine = db.create_engine(
            "mysql+pymysql://%s:%s@%s:%s/%s" % (username, password, ip, port, database_name))
        self.Session = sessionmaker(bind=inner_engine)
        self.session = self.Session()

    def get_all(self):
        """
        Returns Image Table in List Form
        List of (ID, Name, URL, Caption)
        """
        result = self.session.query(self.Image.ID, self.Image.name, self.Image.image_url, self.Image.caption).all()
        return result

    def doesImageNameExist(self, name):
        """
        Returns Count of Name
        Used to check if name exists
        """
        result = self.session.query(self.Image.name).filter(self.Image.name == name).count()
        return result

    def insertImage(self, name, image_url):
        """
        Insert Image, returns ID of image
        """
        self.session.add(self.Image(image_url=image_url, name=name))
        self.session.commit()
        result = self.session.query(self.Image.ID).filter(
            self.Image.name == name).one()
        return result

    def getImageID(self, name):
        """
        No idea where this is used
        """
        result = self.session.query(self.Image.ID).filter(
            self.Image.name == name).one()
        return result

    def findById(self, ID):
        """
        Returns Image Name, URL, Caption
        """
        result = self.session.query(self.Image.name, self.Image.image_url, self.Image.caption).filter(
            self.Image.ID == ID).one()
        return result

    def updateImageURL(self, ID, new_URL):
        try:
            statement = select(self.Image).where(self.Image.ID == ID)
            retrievedImage = self.session.scalars(statement).one()
            retrievedImage.image_url = new_URL
            self.session.commit()
        except Exception as e:
            print(e)

    def deleteImageById(self, ID):
        statement = self.session.query(self.Image).filter(self.Image.ID == ID).delete()
        self.session.commit()

    def updatecaption(self, ID, newcaption):
        try:
            statement = select(self.Image).where(self.Image.ID == ID)
            retrievedcaption = self.session.scalars(statement).one()
            retrievedcaption.caption = newcaption
            self.session.commit()
        except Exception as e:
            print(e)


class QuestionAnsSQL:
    Base = declarative_base()

    class QuestionAnswer(Base):
        __tablename__ = 'QuestionAnswer'
        ID = db.Column(db.INTEGER)
        questionID = db.Column(db.INTEGER, primary_key=True, autoincrement=True)
        question = db.Column(db.VARCHAR(255))
        answer = db.Column(db.VARCHAR(255))

    def __init__(self, username, password, ip, port, database_name):
        inner_engine = db.create_engine(
            "mysql+pymysql://%s:%s@%s:%s/%s" % (username, password, ip, port, database_name))
        self.Session = sessionmaker(bind=inner_engine)
        self.session = self.Session()

    def getDataByID(self, ID):
        """
        Returns list of list? for matching ID
        Format is (questionID, question, answer)
        """
        result = self.session.query(self.QuestionAnswer.questionID, self.QuestionAnswer.question,
                                    self.QuestionAnswer.answer).filter(
            self.QuestionAnswer.ID == ID).all()
        return result

    def updateQuestions(self, QuestionID, NewQuestion):
        """
        Search by QuestionID, Updates using NewQuestion
        """
        try:
            statement = select(self.QuestionAnswer).where(self.QuestionAnswer.questionID == QuestionID)
            retrievedImage = self.session.scalars(statement).one()
            retrievedImage.question = NewQuestion
            self.session.commit()
        except Exception as e:
            return e

    def updateAnswers(self, QuestionID, NewAnswer):
        """
        Search by QuestionID, Updates using NewAnswer
        """
        try:
            statement = select(self.QuestionAnswer).where(self.QuestionAnswer.questionID == QuestionID)
            retrievedImage = self.session.scalars(statement).one()
            retrievedImage.answer = NewAnswer
            self.session.commit()
        except Exception as e:
            return e

    def delete_QuestionAnswer(self, QuestionID):
        """
        Deletes by QuestionID
        """
        try:
            statement = self.session.query(self.QuestionAnswer).filter(
                self.QuestionAnswer.questionID == QuestionID).delete()
            self.session.commit()
        except Exception as e:
            return e

    def insertQuestion(self, ID, Question):
        """
    Insert Question, returns generated QuestionID
        """
        self.session.add(self.QuestionAnswer(ID=ID, question=Question, answer=""))
        self.session.commit()
        result = self.session.query(self.QuestionAnswer.questionID) \
            .filter(self.QuestionAnswer.ID == ID, self.QuestionAnswer.question == Question) \
            .one()
        return result
