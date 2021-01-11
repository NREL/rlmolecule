from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, func

from rlmolecule.sql import Base


class Reward(Base):
    __tablename__ = 'Reward'

    hash = Column(Integer, primary_key=True)
    state = Column(String, primary_key=True)
    run_id = Column(String)
    time = Column(DateTime, server_default=func.now())
    reward = Column(Float)
    data = Column(JSON)


class Game(Base):
    __tablename__ = 'Game'

    index = Column(Integer, primary_key=True)
    id = Column(String)
    run_id = Column(String)
    time = Column(DateTime, server_default=func.now())
    raw_reward = Column(Float)
    scaled_reward = Column(Float)
    search_statistics = Column(JSON)
