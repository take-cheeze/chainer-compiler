#include "graph.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <queue>
#include <set>

#include <onnx/onnx_pb.h>

#include <common/log.h>
#include <compiler/node.h>
#include <compiler/serializer_util.h>
#include <compiler/tensor.h>
#include <compiler/value.h>

namespace oniku {

Graph::Graph(const onnx::GraphProto& xgraph) : name_(xgraph.name()), doc_string_(xgraph.doc_string()) {
    std::map<std::string, Value*> values_by_name;
    for (const onnx::ValueInfoProto& input : xgraph.input()) {
        Value* value = new Value(input, Value::Kind::kInput);
        all_values_.emplace_back(value);
        input_values_.push_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }
    for (const onnx::ValueInfoProto& output : xgraph.output()) {
        Value* value = new Value(output, Value::Kind::kOutput);
        all_values_.emplace_back(value);
        output_values_.push_back(value);
        auto p = values_by_name.emplace(value->name(), value);
        if (!p.second) {
            AddNode(Node::kIdentity, {p.first->second}, {value});
        }
    }
    for (const onnx::ValueInfoProto& temp : xgraph.value_info()) {
        Value* value = new Value(temp, Value::Kind::kTemp);
        all_values_.emplace_back(value);
        temp_values_.push_back(value);
        CHECK(values_by_name.emplace(value->name(), value).second) << "Duplicated value name: " << value->name();
    }

    for (const onnx::TensorProto& xtensor : xgraph.initializer()) {
        std::unique_ptr<Tensor> tensor(new Tensor(xtensor));
        auto found = values_by_name.find(tensor->name());
        CHECK(found != values_by_name.end()) << "Invalid name for an initializer: " << tensor->name();
        CHECK_EQ(found->second->kind(), Value::Kind::kInput) << "Only input can have an initializer but " << found->second->kind();
        found->second->ResetInitializer(std::move(tensor));
    }

    auto get_value = [&](const std::string& name) {
        auto p = values_by_name.emplace(name, nullptr);
        if (!p.second) return p.first->second;
        return p.first->second = AddValue(name);
    };

    for (const onnx::NodeProto& xnode : xgraph.node()) {
        std::vector<Value*> inputs;
        for (const std::string& name : xnode.input()) {
            inputs.push_back(get_value(name));
        }
        std::vector<Value*> outputs;
        for (const std::string& name : xnode.output()) {
            outputs.push_back(get_value(name));
        }

        Node* node = new Node(xnode, inputs, outputs);
        AddNodeImpl(std::unique_ptr<Node>(node), inputs, outputs);
    }
}

Graph::Graph(const std::string name) : name_(name) {
}

Graph::~Graph() {
}

void Graph::ToONNX(onnx::GraphProto* xgraph, bool serialize_initializers) const {
    DUMP_STRING(xgraph, name);
    DUMP_STRING(xgraph, doc_string);

    // `temp_values_` may contain kInput and kOutput due to `ChangeKind`.
    // TODO(hamaji): Remove `ChangeKind` by refactoring how gradients
    // of subgraphs are created.
    std::set<Value*> input_values{input_values_.begin(), input_values_.end()};
    std::set<Value*> output_values{output_values_.begin(), output_values_.end()};
    for (const auto& value : all_values_) {
        onnx::ValueInfoProto* xvalue = nullptr;
        if (input_values.count(value.get())) {
            xvalue = xgraph->add_input();
        } else if (output_values.count(value.get())) {
            xvalue = xgraph->add_output();
        } else if (value->kind() != Value::Kind::kNull) {
            xvalue = xgraph->add_value_info();
        }
        if (!xvalue) continue;

        value->ToONNX(xvalue);
        if (serialize_initializers) {
            if (const Tensor* initializer = value->initializer()) {
                onnx::TensorProto* xtensor = xgraph->add_initializer();
                initializer->ToONNX(xtensor);
            }
        }
    }

    for (const Node* node : nodes_) {
        onnx::NodeProto* xnode = xgraph->add_node();
        node->ToONNX(xnode);
    }
}

std::string Graph::DebugString() const {
    onnx::GraphProto xgraph;
    ToONNX(&xgraph);
    return xgraph.DebugString();
}

std::vector<Node*> Graph::GetLiveNodes() const {
    std::vector<Node*> nodes;
    for (Node* node : nodes_) {
        if (!node->detached()) nodes.push_back(node);
    }
    return nodes;
}

std::set<Value*> Graph::GetNecessaryValues(const std::vector<Value*>& output_values) const {
    std::queue<Value*> q;
    for (Value* value : output_values) q.push(value);

    std::set<Value*> seen_values;
    while (!q.empty()) {
        Value* value = q.front();
        q.pop();
        if (Node* node = value->producer()) {
            for (Value* input : node->inputs()) {
                if (!seen_values.emplace(input).second) continue;
                q.push(input);
            }
        }
    }
    return seen_values;
}

std::set<Value*> Graph::GetNecessaryValues() const {
    return GetNecessaryValues(output_values_);
}

Value* Graph::AddValue(const std::string& name, const Type& type, Value::Kind kind) {
    if (name == "" && kind != Value::Kind::kNull) {
        CHECK_EQ(kind, Value::Kind::kTemp);
        kind = Value::Kind::kNull;
    }
    Value* value = new Value(name, type, kind);
    all_values_.emplace_back(value);
    switch (kind) {
        case Value::Kind::kInput:
            input_values_.push_back(value);
            break;
        case Value::Kind::kOutput:
            output_values_.push_back(value);
            break;
        case Value::Kind::kTemp:
            temp_values_.push_back(value);
            break;
        case Value::Kind::kNull:
            break;
        default:
            CHECK(false) << kind;
    }
    return value;
}

Value* Graph::AddValue(const std::string& name, Value::Kind kind) {
    return AddValue(name, Type(), kind);
}

Value* Graph::AddInputValue(const std::string& name, const Type& type) {
    Value* value = new Value(name, type, Value::Kind::kInput);
    all_values_.emplace_back(value);
    input_values_.push_back(value);
    return value;
}

Value* Graph::AddOutputValue(const std::string& name, const Type& type) {
    Value* value = new Value(name, type, Value::Kind::kOutput);
    all_values_.emplace_back(value);
    output_values_.push_back(value);
    return value;
}

Value* Graph::AddNullValue() {
    return AddValue("", Value::Kind::kNull);
}

Node* Graph::AddNode(Node::OpType op_type, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs, const std::string& base) {
    Node* node = new Node(GenSym(base.empty() ? Node::OpTypeToString(op_type) : base), op_type, inputs, outputs);
    // Node* node = new Node(GenSym(op_type), op_type, inputs, outputs);
    AddNodeImpl(std::unique_ptr<Node>(node), inputs, outputs);
    return node;
}

void Graph::DetachNode(Node* node) {
    node->Detach();
}

std::vector<Node*> Graph::GetTopologicallySortedNodes() const {
    // TODO(hamaji): Add a test for this function.
    std::queue<Value*> q;
    for (Value* value : input_values()) {
        q.push(value);
    }
    std::map<Node*, int> input_counts;
    for (Node* node : GetLiveNodes()) {
        input_counts[node] = node->GetNumActualInputs();
    }

    for (const auto& p : input_counts) {
        if (p.second == 0) {
            for (Value* v : p.first->outputs()) {
                q.push(v);
            }
        }
    }

    std::vector<Node*> sorted_nodes;
    while (!q.empty()) {
        Value* v = q.front();
        q.pop();
        for (Node* node : v->users()) {
            auto found = input_counts.find(node);
            CHECK(found != input_counts.end());
            if (--found->second == 0) {
                sorted_nodes.push_back(node);
                for (Value* n : node->outputs()) {
                    q.push(n);
                }
            }
        }
    }
    return sorted_nodes;
}

std::map<Node*, int> Graph::GetNecessaryNodesAndInputCounts(const std::vector<Value*>& output_values) const {
    std::queue<Node*> q;
    for (const Value* value : output_values) {
        q.push(value->producer());
    }

    // All node in this graph for sanity check.
    std::set<Node*> node_set(nodes_.begin(), nodes_.end());

    std::map<Node*, int> input_counts;
    while (!q.empty()) {
        Node* node = q.front();
        q.pop();
        if (!node) continue;
        if (!input_counts.emplace(node, node->GetNumActualInputs()).second) continue;
        if (!node_set.count(node)) {
            std::cerr << "External reference from " << this->name() << ". External node:\n" << node->DebugString();
            DumpONNXOnFailure();
            CHECK(false);
        }

        for (const Value* input : node->inputs()) {
            q.push(input->producer());
            for (Node* node : input->users()) {
                if (node->outputs().empty()) q.push(node);
            }
        }

        // Nodes without any outputs are always necessary (e.g., OnikuxPrint).
        for (const Value* output : node->outputs()) {
            for (Node* node : output->users()) {
                if (node->outputs().empty()) q.push(node);
            }
        }
    }
    return input_counts;
}

std::vector<const Node*> Graph::GetComputationSequence() const {
    std::vector<const Node*> nodes;
    for (const Node* node : nodes_) {
        if (node->onikux_order() >= 0) nodes.push_back(node);
    }
    std::sort(nodes.begin(), nodes.end(), [](const Node* a, const Node* b) { return a->onikux_order() < b->onikux_order(); });
    return nodes;
}

std::string Graph::GenSym(const std::string& base) {
    std::ostringstream oss;
    if (!base.empty()) oss << base << "_";
    oss << "oniku_gensym_" << ++gen_id_;
    return oss.str();
}

void Graph::AddNodeImpl(std::unique_ptr<Node> node, const std::vector<Value*>& inputs, const std::vector<Value*>& outputs) {
    for (Value* input : inputs) input->AddUser(node.get());
    for (Value* output : outputs) output->SetProducer(node.get());
    nodes_.push_back(node.get());
    nodes_buf_.emplace_back(std::move(node));
}

void Graph::MigrateNodes(const std::vector<Node*>& nodes, const std::vector<Value*> temps, Graph* to) {
    for (Node* node : nodes) {
        auto found = std::find(nodes_.begin(), nodes_.end(), node);
        CHECK(found != nodes_.end()) << node->DebugString();
        nodes_.erase(found);
        to->nodes_.push_back(node);
    }
    for (Value* value : temps) {
        auto found = std::find(temp_values_.begin(), temp_values_.end(), value);
        CHECK(found != temp_values_.end()) << value->DebugString();
        temp_values_.erase(found);
        to->temp_values_.push_back(value);
    }
}

void Graph::ResetGradients() {
    for (const auto& v : all_values()) {
        if (Value* gv = v->grad()) {
            gv->set_type(new Type(v->type()));
            v->set_grad(nullptr);
        }
    }
}

void Graph::DeleteDetached() {
    nodes_ = GetLiveNodes();
}

void Graph::DumpSubGraphs(int depth) const {
    for (int i = 0; i < depth; i++) std::cerr << ' ';
    std::cerr << name() << ' ' << input_values().size() << " inputs " << output_values().size() << " outputs" << std::endl;
    for (const Node* node : nodes_) {
        for (Graph* sub_graph : node->GetSubGraphs()) {
            sub_graph->DumpSubGraphs(depth + 1);
        }
    }
}

void Graph::DumpONNXOnFailure(const std::string& filename) const {
    onnx::ModelProto xmodel;
    xmodel.set_ir_version(3);
    xmodel.set_producer_name("oniku failed :(");
    ToONNX(xmodel.mutable_graph());
    const std::string fn = filename.empty() ? "/tmp/oniku_failure.onnx" : filename;
    std::ofstream ofs(fn);
    xmodel.SerializeToOstream(&ofs);
    std::cerr << "Failed graph is stored in " << fn << std::endl;
}

}  // namespace oniku
