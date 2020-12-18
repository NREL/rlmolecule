class MoleculeTreeNode(object):
    @property
    def policy_inputs(self):
        """
        :return GNN inputs for the node
        """
        if self._policy_inputs is None:
            self._policy_inputs = self._game.construct_feature_matrices(self)
        return self._policy_inputs

    def policy_inputs_with_children(self) -> {}:
        """
        :return the given nodes policy inputs, concatenated together with the
        inputs of its successor nodes. Used as the inputs for the policy neural
        network
        """

        policy_inputs = [node.policy_inputs for node in itertools.chain((self,), self.get_successors())]
        return {key: pad_sequences([elem[key] for elem in policy_inputs], padding='post')
                for key in policy_inputs[0].keys()}

    def store_policy_data(self):
        data = self.policy_inputs_with_children()
        visit_counts = np.array([child.visits for child in self.get_successors()])
        data['visit_probs'] = visit_counts / visit_counts.sum()

        with io.BytesIO() as f:
            np.savez_compressed(f, **data)
            self._policy_data = f.getvalue()

    @property
    def policy_data(self):
        if self._policy_data is None:
            self.store_policy_data()
        return self._policy_data
