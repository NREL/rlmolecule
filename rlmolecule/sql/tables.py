from sqlalchemy import BigInteger, Column, DateTime, Float, Index, Integer, JSON, String, UniqueConstraint, func

from rlmolecule.sql import Base


class RewardStore(Base):
    __tablename__ = 'Reward'

    index = Column(Integer, primary_key=True)
    hash = Column(BigInteger)
    run_id = Column(String)
    state = Column(String)
    time = Column(DateTime, server_default=func.now())
    reward = Column(Float)
    data = Column(JSON)
    __table_args__ = (UniqueConstraint('state', 'run_id', name='_state_runid_uc'),
                      Index('hash', 'run_id'),
                      )


class GameStore(Base):
    __tablename__ = 'Game'

    index = Column(Integer, primary_key=True)
    id = Column(String)
    run_id = Column(String)
    time = Column(DateTime, server_default=func.now())
    raw_reward = Column(Float)
    scaled_reward = Column(Float)
    search_statistics = Column(JSON)
