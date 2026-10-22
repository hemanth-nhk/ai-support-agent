import io
import json
import uuid
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.analytics.dataset_service import ingest_dataset, get_dataset
from app.analytics.planner import generate_analysis_plan, plan_mutation
from app.analytics.compiler import compile_plan
from app.analytics.executor import execute_compiled_query
from app.analytics.visualizer import render_chart
from app.analytics.schemas import PlanMutationType

client = TestClient(app)


def test_semicolon_delimited_healthcare_dataset():
    """
    Evaluates unfamiliar snake_case schema, semicolon delimiters, nulls,
    and floating-point values in a healthcare dataset.
    """
    session_id = f"test_session_hc_{uuid.uuid4().hex[:8]}"
    csv_content = (
        "patient_id;clinic_ward;readmission_risk;treatment_cost_usd\n"
        "P101;Cardiology;0.82;14500.50\n"
        "P102;Neurology;;9200.00\n"
        "P103;Cardiology;0.45;12100.25\n"
        "P104;Oncology;0.91;28400.00\n"
        "P105;Neurology;0.15;4300.75\n"
        "P106;Cardiology;0.65;16800.00\n"
        "P107;Oncology;;31200.50\n"
    ).encode("utf-8")

    # 1. Ingest and profile
    res = ingest_dataset(session_id, "healthcare_records.csv", csv_content)
    assert res.dataset_id is not None
    assert res.profile.row_count == 7
    col_names = [c.name for c in res.profile.columns]
    assert "clinic_ward" in col_names
    assert "treatment_cost_usd" in col_names
    assert "readmission_risk" in col_names

    # Check null handling in profiler
    risk_col = next(c for c in res.profile.columns if c.name == "readmission_risk")
    assert risk_col.missing_count == 2

    # 2. Plan aggregation by clinic_ward
    metadata = get_dataset(res.dataset_id, session_id=session_id)
    plan1 = generate_analysis_plan(
        "Total treatment cost by clinic ward",
        metadata=metadata,
    )
    assert plan1.group_by[0].column == "clinic_ward"
    assert plan1.aggregations[0].column == "treatment_cost_usd"

    # 3. Compile and execute
    compiled1 = compile_plan(plan1, metadata)
    result1 = execute_compiled_query(compiled1, res.dataset_id, session_id, plan1)
    agg_alias = plan1.aggregations[0].alias or "sum_treatment_cost_usd"
    assert agg_alias in result1.columns

    # 4. Multi-turn: Filter & Sort mutation
    mutation = plan_mutation(
        "Filter to Cardiology",
        metadata=metadata,
        previous_plan=plan1,
    )
    assert mutation.mutation_type in (PlanMutationType.ADD_FILTER, PlanMutationType.CHANGE_FILTER)

    # 5. Chart generation
    chart_artifact = render_chart(result1, "bar", session_id, title="Treatment Cost by Ward")
    assert chart_artifact.chart_id is not None
    assert chart_artifact.filename is not None
    assert chart_artifact.url.startswith("/charts/")

    # 6. Export validation (CSV/JSON serializable structure)
    assert len(result1.columns) >= 2
    assert len(result1.rows) == 3
    json_export = json.dumps(result1.rows)
    assert "Cardiology" in json_export
    csv_rows = [",".join(str(v) for v in row.values()) for row in result1.rows]
    assert any("Cardiology" in r for r in csv_rows)


def test_tab_delimited_telemetry_tsv():
    """
    Evaluates tab-delimited telemetry dataset with duplicate rows, mixed types,
    and server datacenter zones.
    """
    session_id = f"test_session_tel_{uuid.uuid4().hex[:8]}"
    tsv_content = (
        "node_name\tdatacenter_zone\tcpu_utilization_pct\tmemory_mb\n"
        "worker-01\tus-east-1a\t78.5\t16384\n"
        "worker-02\tus-east-1b\t45.2\t8192\n"
        "worker-03\tus-east-1a\t91.0\t32768\n"
        "worker-04\teu-central-1a\t63.4\t16384\n"
        "worker-01\tus-east-1a\t78.5\t16384\n"  # Duplicate row
        "worker-05\teu-central-1b\t88.1\t32768\n"
    ).encode("utf-8")

    res = ingest_dataset(session_id, "node_telemetry.tsv.csv", tsv_content)
    assert res.dataset_id is not None
    assert res.profile.row_count == 6

    metadata = get_dataset(res.dataset_id, session_id=session_id)
    plan = generate_analysis_plan(
        "Average cpu utilization by datacenter zone",
        metadata=metadata,
    )
    assert plan.group_by[0].column == "datacenter_zone"

    compiled = compile_plan(plan, metadata)
    result = execute_compiled_query(compiled, res.dataset_id, session_id, plan)
    assert len(result.rows) >= 3


def test_excel_xlsx_logistics_dataset():
    """
    Evaluates binary Excel .xlsx upload, openpyxl engine parsing, null values,
    and ranking logic.
    """
    session_id = f"test_session_xlsx_{uuid.uuid4().hex[:8]}"
    df = pd.DataFrame([
        {"warehouse_code": "WH-NORTH", "transport_mode": "Air", "shipment_cost": 4200.0, "carrier_rating": 4.8},
        {"warehouse_code": "WH-SOUTH", "transport_mode": "Ground", "shipment_cost": 1500.0, "carrier_rating": 3.9},
        {"warehouse_code": "WH-EAST", "transport_mode": "Sea", "shipment_cost": 890.0, "carrier_rating": 4.2},
        {"warehouse_code": "WH-WEST", "transport_mode": "Air", "shipment_cost": 3800.0, "carrier_rating": None},
        {"warehouse_code": "WH-NORTH", "transport_mode": "Ground", "shipment_cost": 1200.0, "carrier_rating": 4.5},
    ])
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    xlsx_bytes = buf.getvalue()

    res = ingest_dataset(session_id, "freight_logistics.xlsx", xlsx_bytes)
    assert res.dataset_id is not None
    assert res.profile.row_count == 5

    metadata = get_dataset(res.dataset_id, session_id=session_id)
    assert metadata.format == "xlsx"

    plan = generate_analysis_plan(
        "Top 2 warehouses by shipment cost",
        metadata=metadata,
    )
    assert plan.limit == 2
    assert plan.order_by[0].descending is True

    compiled = compile_plan(plan, metadata)
    result = execute_compiled_query(compiled, res.dataset_id, session_id, plan)
    assert len(result.rows) == 2


def test_json_records_dataset():
    """
    Evaluates JSON records format upload, column discovery, and multi-turn
    order reversal.
    """
    session_id = f"test_session_json_{uuid.uuid4().hex[:8]}"
    data = [
        {"customer_tier": "Enterprise", "region": "APAC", "nps_score": 9, "mrr": 45000},
        {"customer_tier": "Mid-Market", "region": "EMEA", "nps_score": 7, "mrr": 12000},
        {"customer_tier": "Enterprise", "region": "AMER", "nps_score": 10, "mrr": 82000},
        {"customer_tier": "SMB", "region": "AMER", "nps_score": 6, "mrr": 3500},
        {"customer_tier": "Enterprise", "region": "EMEA", "nps_score": 8, "mrr": 67000},
    ]
    json_bytes = json.dumps(data).encode("utf-8")

    res = ingest_dataset(session_id, "customer_health.json", json_bytes)
    assert res.dataset_id is not None
    assert res.profile.row_count == 5

    metadata = get_dataset(res.dataset_id, session_id=session_id)
    plan1 = generate_analysis_plan(
        "Top customer tiers by mrr",
        metadata=metadata,
    )
    assert plan1.order_by[0].descending is True

    # Mutate: reverse order
    plan2 = generate_analysis_plan(
        "Reverse the order",
        metadata=metadata,
        previous_plan=plan1,
    )
    assert plan2.order_by[0].descending is False


def test_dataset_switching_and_state_isolation():
    """
    Verifies that uploading and analyzing Dataset A does not leak column schemas,
    filters, or state into Dataset B, and switching back restores isolation.
    """
    session_id = f"test_session_iso_{uuid.uuid4().hex[:8]}"

    # Dataset A: Finance
    ds_a_csv = (
        "account_id,department,spend_usd\n"
        "A1,Marketing,5000\n"
        "A2,Sales,8000\n"
    ).encode("utf-8")
    res_a = ingest_dataset(session_id, "finance.csv", ds_a_csv)

    # Dataset B: Fleet
    ds_b_csv = (
        "vehicle_id,fuel_type,mileage_km\n"
        "V10,Electric,12000\n"
        "V11,Hybrid,34000\n"
    ).encode("utf-8")
    res_b = ingest_dataset(session_id, "fleet.csv", ds_b_csv)

    meta_a = get_dataset(res_a.dataset_id, session_id=session_id)
    meta_b = get_dataset(res_b.dataset_id, session_id=session_id)

    # Analyze Dataset A
    plan_a = generate_analysis_plan("Total spend by department", metadata=meta_a)
    assert plan_a.dataset_id == res_a.dataset_id
    assert plan_a.group_by[0].column == "department"

    # Analyze Dataset B - must not reference department or spend_usd
    plan_b = generate_analysis_plan("Average mileage by fuel type", metadata=meta_b)
    assert plan_b.dataset_id == res_b.dataset_id
    assert plan_b.group_by[0].column == "fuel_type"
    assert "department" not in [g.column for g in plan_b.group_by]
    assert "spend_usd" not in [a.column for a in plan_b.aggregations]

    # Verify DuckDB execution isolation
    comp_a = compile_plan(plan_a, meta_a)
    res_exec_a = execute_compiled_query(comp_a, res_a.dataset_id, session_id, plan_a)
    assert "department" in res_exec_a.columns

    comp_b = compile_plan(plan_b, meta_b)
    res_exec_b = execute_compiled_query(comp_b, res_b.dataset_id, session_id, plan_b)
    assert "fuel_type" in res_exec_b.columns
