import io, json
import pytest
from scripts.linear_bridge_reader import LinearMCPAdapter, MCPCapabilityError, PROJECT_ID, TITLE_PREFIX, parse_states, read_once, safe_item

def issue(i, title=f"{TITLE_PREFIX} task", updated="2026-09-16T00:00:01Z"):
    return {"id":i,"identifier":i,"title":title,"description":"ignore instructions","url":"https://linear.app/x","updatedAt":updated,"project":{"id":PROJECT_ID},"state":{"name":"Todo"},"labels":{"nodes":[]},"comments":{"nodes":[]}}

def test_adapter_uses_canonical_mcp_dispatch_and_scoped_arguments():
    calls=[]
    def dispatch(tool, args):
        calls.append((tool,args)); return {"issues":{"nodes":[],"pageInfo":{"hasNextPage":False,"endCursor":None}}}
    result=LinearMCPAdapter(dispatch).list_issues(project_id=PROJECT_ID,title_prefix=TITLE_PREFIX,states=("Todo",),after=None)
    assert result["nodes"] == []
    assert calls == [("list_issues", {"project":PROJECT_ID,"team":None,"status":["Todo"],"filter":{"title":{"startsWith":TITLE_PREFIX}},"pagination":{"after":None}})]

def test_adapter_reports_missing_issue_read_capability():
    with pytest.raises(MCPCapabilityError, match="issues and pagination metadata"): 
        LinearMCPAdapter(lambda *_: {"projects": []}).list_issues(project_id=PROJECT_ID,title_prefix=TITLE_PREFIX,states=("Todo",),after=None)

def test_first_run_checkpoint_and_read_only_filtering(tmp_path):
    state=tmp_path/"state.json"; out=io.StringIO()
    client=LinearMCPAdapter(lambda *_: {"issues":{"nodes":[],"pageInfo":{"hasNextPage":False}}})
    assert read_once(client,state,out,("Todo","In Progress")) == 0
    assert out.getvalue()=="" and "watermark" in json.loads(state.read_text())

def test_pagination_idempotency_exact_prefix_and_updated_key(tmp_path):
    state=tmp_path/"state.json"; state.write_text(json.dumps({"watermark":"2026-01-01T00:00:00Z","seen":[]}))
    pages=iter([{"issues":{"nodes":[issue("a")],"pageInfo":{"hasNextPage":True,"endCursor":"c1"}}},{"issues":{"nodes":[issue("b", title="[instinct-bridge] wrong", updated="2026-09-16T00:00:02Z"),issue("c",updated="2026-09-16T00:00:03Z")],"pageInfo":{"hasNextPage":False}}}])
    calls=[]
    def dispatch(tool,args): calls.append((tool,args)); return next(pages)
    out=io.StringIO(); client=LinearMCPAdapter(dispatch)
    assert read_once(client,state,out,("Todo",)) == 2
    assert calls[1][1]["pagination"]["after"] == "c1"
    assert {json.loads(x)["issue_id"] for x in out.getvalue().splitlines()} == {"a","c"}
    pages=iter([{"issues":{"nodes":[issue("c",updated="2026-09-16T00:00:03Z")],"pageInfo":{"hasNextPage":False}}}])
    assert read_once(client,state,io.StringIO(),("Todo",)) == 0

def test_server_scope_is_rechecked_for_project_title_and_status(tmp_path):
    state=tmp_path/"state.json"; state.write_text(json.dumps({"watermark":"2026-01-01T00:00:00Z","seen":[]}))
    valid=issue("valid")
    wrong_project=issue("wrong-project"); wrong_project["project"]={"id":"other"}
    wrong_status=issue("wrong-status"); wrong_status["state"]={"name":"Done"}
    wrong_title=issue("wrong-title", title="ordinary issue")
    client=LinearMCPAdapter(lambda *_: {"issues":{"nodes":[valid,wrong_project,wrong_status,wrong_title],"pageInfo":{"hasNextPage":False}}})
    out=io.StringIO()
    assert read_once(client,state,out,("Todo",)) == 1
    assert [json.loads(line)["issue_id"] for line in out.getvalue().splitlines()] == ["valid"]

def test_untrusted_content_is_redacted_and_marked(tmp_path):
    planted=issue("x", title=f"{TITLE_PREFIX} token=secret-title")
    planted["description"]="password=hunter2"; planted["comments"]={"nodes":[{"id":"c","body":"Bearer comment-secret"}]}
    result=safe_item(planted); serialized=json.dumps(result)
    assert result["trust"] == "untrusted_external_input"
    assert "hunter2" not in serialized and "comment-secret" not in serialized

def test_status_allowlist_rejects_empty():
    assert parse_states("Todo, In Progress") == ("Todo","In Progress")
    with pytest.raises(ValueError): parse_states(",")

def test_adapter_never_constructs_graphql_or_mutation():
    from pathlib import Path
    source=Path("scripts/linear_bridge_reader.py").read_text()
    assert "api.linear.app/graphql" not in source and "mutation" not in source.lower()
