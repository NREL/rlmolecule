from sqlalchemy import (BigInteger, Column, DateTime, Float, JSON, LargeBinary, String,
                        func)

from rlmolecule.sql import Base


class RewardStore(Base):
    __tablename__ = 'Reward'

    digest = Column(String(128), primary_key=True)
    hash = Column(BigInteger, primary_key=True)
    run_id = Column(String(255), primary_key=True)
    time = Column(DateTime, server_default=func.now())
    reward = Column(Float)
    data = Column(JSON)
    

class StateStore(Base):
    __tablename__ = 'State'

    digest = Column(String(128), primary_key=True)
    hash = Column(BigInteger, primary_key=True)
    run_id = Column(String(255), primary_key=True)
    state = Column(LargeBinary)
    policy_inputs = Column(LargeBinary)


class GameStore(Base):
    __tablename__ = 'Game'

    id = Column(String(64), primary_key=True)  # UUID
    run_id = Column(String(255))
    time = Column(DateTime, server_default=func.now())
    raw_reward = Column(Float)
    scaled_reward = Column(Float)
    search_statistics = Column(JSON)
