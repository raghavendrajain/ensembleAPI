

# Helper for the session
def get_variables(acc_variables, all_variables, all_models_names, model_name):
  variables_to_restore = {}
  for variable in all_variables:
    base_scope = variable.op.name.split("/", 1)
    
    if base_scope[0] in all_models_names:
      if base_scope[0] == model_name:
        variables_to_restore[base_scope[1]] = variable
    else:
      if base_scope[0] in acc_variables:
        pass
      elif '_'.join(base_scope[0].split("_")[:-1]) in acc_variables:
        name_in_ckpt = '/'.join(['_'.join(base_scope[0].split("_")[:-1]), base_scope[1]])
        variables_to_restore[name_in_ckpt] = variable
      else:
        acc_variables.append(base_scope[0])
        variables_to_restore[variable.op.name] = variable

  return acc_variables, variables_to_restore


# True if no duplication over the list
def allUnique(x):
  seen = set()
  return not any(i in seen or seen.add(i) for i in x)