"""Atomic local OpenAI budget ledger, including failed and streamed requests."""
import json,os,sqlite3,time,uuid
from pathlib import Path
import httpx
from contextlib import contextmanager
from contextvars import ContextVar

_CALL_STAGE = ContextVar("refmind_call_stage", default=None)

@contextmanager
def billing_stage(name):
    token = _CALL_STAGE.set(name)
    try:
        yield
    finally:
        _CALL_STAGE.reset(token)


class BudgetExceeded(RuntimeError): pass

def ledger_path():
    return Path(os.getenv("OPENAI_USAGE_DB", str(Path(__file__).resolve().parents[2]/"data/openai_usage.sqlite3")))

@contextmanager
def connect():
    p=ledger_path();p.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(p,timeout=30)
    c.execute("CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY, started REAL, day TEXT, model TEXT, role TEXT, reserved REAL, cost REAL, status TEXT, input_tokens INTEGER, output_tokens INTEGER, elapsed REAL)")
    try:
        yield c
        c.commit()
    except BaseException:
        c.rollback()
        raise
    finally:
        c.close()

class MeteredTransport(httpx.BaseTransport):
    def __init__(self,role):
        self.role=role
        self.inner=httpx.HTTPTransport(proxy=os.getenv("OPENAI_PROXY") or None,retries=0)
    def handle_request(self,request):
        payload=json.loads(request.content)
        model=payload.get("model","")
        embedding=model == "text-embedding-3-small"
        if model not in {"text-embedding-3-small", "gpt-5.4-mini", "gpt-5.4-mini-2026-03-17"}:
            raise ValueError("No verified price/budget profile for model")
        inp_rate,out_rate=(.02,0) if embedding else (.75,4.5)
        # UTF-8 bytes are a conservative text token bound; include the output cap.
        reserve=len(request.content)*inp_rate/1e6 + int(payload.get("max_completion_tokens",4096))*out_rate/1e6+.01
        if any("image_url" in str(m) for m in payload.get("messages",[])): reserve+=1
        call_id=uuid.uuid4().hex;started=time.time()
        from datetime import datetime
        from zoneinfo import ZoneInfo
        day=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        with connect() as c:
            c.execute("BEGIN IMMEDIATE")
            total=c.execute("SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM calls").fetchone()[0]
            daily=c.execute("SELECT COALESCE(SUM(COALESCE(cost,reserved)),0) FROM calls WHERE day=?",(day,)).fetchone()[0]
            if total+reserve>float(os.getenv("OPENAI_TOTAL_BUDGET_USD","10")) or daily+reserve>float(os.getenv("OPENAI_DAILY_BUDGET_USD","2")):
                raise BudgetExceeded("OpenAI application budget exhausted")
            c.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,NULL,?,NULL,NULL,NULL)",(call_id,started,day,model,_CALL_STAGE.get() or self.role,reserve,"pending"))
        try:
            response=self.inner.handle_request(request)
            body=response.read()
            usage={}
            try: usage=json.loads(body).get("usage") or {}
            except (ValueError,AttributeError):
                for line in body.splitlines():
                    if line.startswith(b"data: ") and line[6:]!=b"[DONE]":
                        event=json.loads(line[6:]);usage=event.get("usage") or usage
            it=usage.get("prompt_tokens",usage.get("total_tokens"))
            ot=usage.get("completion_tokens",0)
            cost=(it*inp_rate+ot*out_rate)/1e6 if it is not None else None
            with connect() as c:
                c.execute("UPDATE calls SET cost=?,status=?,input_tokens=?,output_tokens=?,elapsed=? WHERE id=?",(cost,"ok" if response.status_code<400 else "http_error",it,ot,time.time()-started,call_id))
            return response
        except Exception:
            with connect() as c: c.execute("UPDATE calls SET status=?,elapsed=? WHERE id=?",("unknown",time.time()-started,call_id))
            raise
    def close(self): self.inner.close()

def client(role):
    return httpx.Client(transport=MeteredTransport(os.getenv("OPENAI_CALL_ROLE",role)),timeout=45)
